#ifndef NEMOTRON_MAX_SHIM_H
#define NEMOTRON_MAX_SHIM_H
#include <stddef.h>

typedef struct NemotronMAX NemotronMAX;
NemotronMAX *nemotron_open(const char *directory, char **error);
void nemotron_close(NemotronMAX *runtime);
int nemotron_embed(NemotronMAX *runtime, const float *features, size_t frames,
                   size_t width, size_t hidden, float *output, char **error);
int nemotron_infer(NemotronMAX *runtime, const float *embeddings,
                   const float *cosines, const float *sines, const float *mask,
                   size_t frames, size_t hidden, size_t rotary,
                   size_t output_count, float *output, char **error);
#endif
