CC = gcc
CFLAGS = -O3 -march=native -fPIC -flto -Wall -Wextra -std=c11 -Iinclude
LDFLAGS = -flto -lm
OMP_CFLAGS = -O3 -march=native -fopenmp -Wall -Wextra -std=c11 -Iinclude
OMP_LDFLAGS = -fopenmp -lm

CXX = g++
CXXFLAGS = -O3 -march=native -std=c++17 -pthread -I/tmp/quackle -Iinclude
QUACKLE_LIB = /tmp/quackle/build/libquackle.a /tmp/quackle/test/trademarkedboards.cpp -lpthread -lm

SRCS = src/dict.c src/board.c src/movegen.c src/eval.c src/nnue.c src/zobrist.c src/palimpsest.c src/lattice.c src/settle.c src/broker.c src/court.c src/simulation.c src/protocol.c src/endgame.c src/bayes_rack.c src/bot_archetypes.c src/quackle_bridge.c
OBJS = $(SRCS:.c=.o)
TARGET = tessera

TEST_SRCS = $(wildcard tests/*.c)
TEST_BINS = $(TEST_SRCS:.c=)

.PHONY: all clean test bench match-bench dataset train zig zig-test zig-bench zig-cross lib tools

all: $(TARGET) lib

tools: tools/match_benchmark tools/generate_dataset tools/train_nnue

lib: libtessera.so

libtessera.so: $(OBJS)
	$(CC) -shared $(OBJS) -o $@ $(LDFLAGS)

$(TARGET): $(OBJS) src/main.o
	$(CC) $(OBJS) src/main.o -o $@ $(LDFLAGS)

HEADERS = $(wildcard include/*.h)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TARGET) $(TEST_BINS)
	@echo "Running test suite..."
	@./tests/test_dict
	@./tests/test_movegen
	@./tests/test_eval
	@./tests/test_incremental_nnue
	@./tests/test_lattice_proof
	@./tests/test_endgame
	@./tests/test_quackle_bridge
	@echo "All tests passed successfully!"

tests/%: tests/%.c $(OBJS)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

bench: $(TARGET)
	@./$(TARGET) bench

match-bench: $(TARGET) tools/match_benchmark
	@./tools/match_benchmark 20

tools/match_benchmark: tools/match_benchmark.c $(OBJS)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

tools/generate_dataset: tools/generate_dataset.c $(OBJS)
	$(CC) $(OMP_CFLAGS) $< $(OBJS) -o $@ $(OMP_LDFLAGS)

tools/train_nnue: tools/train_nnue.c src/nnue.o
	$(CC) $(OMP_CFLAGS) $< src/nnue.o -o $@ $(OMP_LDFLAGS)

dataset: tools/generate_dataset
	@./tools/generate_dataset 12000 data/train_data.bin

train: tools/train_nnue
	@./tools/train_nnue data/train_data.bin data/nnue.bin 25

zig:
	zig build -Doptimize=ReleaseFast

zig-test:
	zig build test -Doptimize=ReleaseFast

zig-bench:
	zig build bench -Doptimize=ReleaseFast

zig-cross:
	@./tools/cross_compile.sh

tools/quackle_benchmark: tools/quackle_benchmark.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $< $(OBJS) $(QUACKLE_LIB) -o $@

clean:
	rm -rf src/*.o tests/*.o tools/match_benchmark tools/generate_dataset tools/train_nnue tools/quackle_benchmark $(TARGET) libtessera.so tests/test_dict tests/test_movegen tests/test_eval tests/test_incremental_nnue tests/test_lattice_proof tests/test_endgame .zig-cache zig-out dist
