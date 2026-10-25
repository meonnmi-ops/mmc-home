CC      ?= gcc
CFLAGS  ?= -I runtime -O2
RUNTIME  = runtime/runtime_v5.c runtime/runtime_glue.c

.PHONY: all verify examples clean

all: build/mmc

build/mmc: bootstrap/compiler_v11_stage0.c $(RUNTIME)
	@mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@ -lm

verify:
	@./scripts/verify_selfhost.sh

examples: build/mmc
	@mkdir -p build/examples
	@for f in examples/*.mmc; do \
		name=$$(basename $$f .mmc); \
		echo "─── $$name ───"; \
		./build/mmc $$f build/examples/$$name.c || continue; \
		$(CC) $(CFLAGS) build/examples/$$name.c $(RUNTIME) -o build/examples/$$name -lm; \
		echo "Run:"; ./build/examples/$$name; echo ""; \
	done

clean:
	rm -rf build
