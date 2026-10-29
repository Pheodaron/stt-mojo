#define _POSIX_C_SOURCE 200809L
#include "shim.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "max/c/common.h"
#include "max/c/context.h"
#include "max/c/device.h"
#include "max/c/model.h"
#include "max/c/safetensors.h"
#include "max/c/tensor.h"
#include "max/c/weights.h"

struct NemotronMAX {
  M_RuntimeConfig *config;
  M_RuntimeContext *context;
  M_Device *device;
  M_Safetensors *weights;
  M_WeightsRegistry *registry;
  M_AsyncCompiledModel *compiled[2];
  M_AsyncModel *models[2];
};

static void set_error(char **error, const char *message) {
  *error = strdup(message ? message : "unknown MAX error");
}

void nemotron_close(NemotronMAX *r) {
  if (!r) return;
  for (int i = 1; i >= 0; --i) {
    if (r->models[i]) M_freeModel(r->models[i]);
    if (r->compiled[i]) M_freeCompiledModel(r->compiled[i]);
  }
  if (r->registry) M_freeWeightsRegistry(r->registry);
  if (r->weights) M_freeSafetensors(r->weights);
  if (r->device) M_freeDevice(r->device);
  if (r->context) M_freeRuntimeContext(r->context);
  if (r->config) M_freeRuntimeConfig(r->config);
  free(r);
}

NemotronMAX *nemotron_open(const char *directory, char **error) {
  NemotronMAX *r = calloc(1, sizeof(*r));
  M_Status *status = M_newStatus();
  char *path = malloc(strlen(directory) + 80);
  if (!r || !path) { set_error(error, "allocating MAX runtime"); goto fail; }
  if (strcmp(M_version(), "26.7.0")) {
    set_error(error, "bundle requires MAX runtime 26.7.0"); goto fail;
  }
#define CHECK_OPEN() if (M_isError(status)) { set_error(error, M_getError(status)); goto fail; }
  r->config = M_newRuntimeConfig();
  r->context = M_newRuntimeContext(r->config, status);
  CHECK_OPEN();
  r->device = M_newDevice(M_HOST, 0, status);
  CHECK_OPEN();
  sprintf(path, "%s/weights.safetensors", directory);
  const char *paths[] = {path};
  r->weights = M_loadSafetensors(paths, 1, r->device, status);
  CHECK_OPEN();
  r->registry = M_newWeightsRegistryFromSafetensors(r->weights, status);
  CHECK_OPEN();
  const char *files[] = {"nemotron_feature_stacking.mef", "nemotron_encoder_and_head.mef"};
  for (int i = 0; i < 2; ++i) {
    sprintf(path, "%s/%s", directory, files[i]);
    M_CompileConfig *compilation = M_newCompileConfig();
    M_setModelPath(compilation, path);
    r->compiled[i] = M_compileModelSync(r->context, &compilation, status);
    if (compilation) M_freeCompileConfig(compilation);
    CHECK_OPEN();
    r->models[i] = M_initModel(r->context, r->compiled[i], r->registry, status);
    CHECK_OPEN();
    M_waitForModel(r->models[i], status);
    CHECK_OPEN();
  }
  free(path);
  M_freeStatus(status);
  return r;
fail:
  free(path);
  M_freeStatus(status);
  nemotron_close(r);
  return NULL;
}

// All borrowing and synchronous execution occur inside a single cgo call.
// No Go pointer is retained by MAX after this function returns.
static int execute(NemotronMAX *r, int model, const float **values,
                   const int64_t shapes[][4], const size_t *ranks, int count,
                   size_t output_count, float *destination, char **error) {
  M_Status *status = M_newStatus();
  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(r->context);
  M_AsyncTensorMap *outputs = NULL;
  M_AsyncTensor *output = NULL;
  M_TensorSpec *specs[4] = {NULL};
  int success = 0;
#define CHECK_RUN() if (M_isError(status)) { set_error(error, M_getError(status)); goto done; }
  for (int i = 0; i < count; ++i) {
    char name[16];
    snprintf(name, sizeof(name), "input%d", i);
    specs[i] = M_newTensorSpec(shapes[i], ranks[i], M_FLOAT32, name, r->device);
    M_borrowTensorInto(inputs, (void *)values[i], specs[i], status);
    CHECK_RUN();
  }
  outputs = M_executeModelSync(r->context, r->models[model], inputs, status);
  CHECK_RUN();
  output = M_getTensorByNameFrom(outputs, "output0", status);
  CHECK_RUN();
  if (M_getTensorType(output) != M_FLOAT32 ||
      M_getTensorNumElements(output) != output_count || !M_getTensorData(output)) {
    set_error(error, "MAX returned an unexpected output tensor"); goto done;
  }
  memcpy(destination, M_getTensorData(output), output_count*sizeof(float));
  success = 1;
done:
  if (output) M_freeTensor(output);
  if (outputs) M_freeAsyncTensorMap(outputs);
  if (inputs) M_freeAsyncTensorMap(inputs);
  for (int i = 0; i < count; ++i) if (specs[i]) M_freeTensorSpec(specs[i]);
  M_freeStatus(status);
  return success;
}

int nemotron_embed(NemotronMAX *r, const float *features, size_t frames,
                   size_t width, size_t hidden, float *output, char **error) {
  const float *values[] = {features};
  const int64_t shapes[][4] = {{1, (int64_t)frames, (int64_t)width, 0}};
  const size_t ranks[] = {3};
  return execute(r, 0, values, shapes, ranks, 1, frames*hidden, output, error);
}

int nemotron_infer(NemotronMAX *r, const float *embeddings,
                   const float *cosines, const float *sines, const float *mask,
                   size_t frames, size_t hidden, size_t rotary,
                   size_t output_count, float *output, char **error) {
  const float *values[] = {embeddings, cosines, sines, mask};
  const int64_t shapes[][4] = {
    {1, (int64_t)frames, (int64_t)hidden, 0},
    {1, 1, (int64_t)frames, (int64_t)rotary},
    {1, 1, (int64_t)frames, (int64_t)rotary},
    {1, 1, 1, (int64_t)frames}
  };
  const size_t ranks[] = {3, 4, 4, 4};
  return execute(r, 1, values, shapes, ranks, 4, output_count, output, error);
}
