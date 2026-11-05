CC = gcc
CFLAGS = -O3 -march=native -flto -Wall -Wextra -std=c11 -Iinclude
LDFLAGS = -flto -lm

SRCS = src/dict.c src/board.c src/movegen.c src/eval.c src/zobrist.c src/palimpsest.c src/lattice.c src/settle.c src/broker.c src/court.c src/simulation.c src/protocol.c
OBJS = $(SRCS:.c=.o)
TARGET = tessera

TEST_SRCS = $(wildcard tests/*.c)
TEST_BINS = $(TEST_SRCS:.c=)

.PHONY: all clean test bench

all: $(TARGET)

$(TARGET): $(OBJS) src/main.o
	$(CC) $(OBJS) src/main.o -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TARGET) $(TEST_BINS)
	@echo "Running test suite..."
	@./tests/test_dict
	@./tests/test_movegen
	@./tests/test_eval
	@./tests/test_lattice_proof
	@echo "All tests passed successfully!"

tests/%: tests/%.c $(OBJS)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

bench: $(TARGET)
	@./$(TARGET) bench

match-bench: $(TARGET) tools/match_benchmark
	@./tools/match_benchmark 20

tools/match_benchmark: tools/match_benchmark.c $(OBJS)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

clean:
	rm -f src/*.o tests/*.o tools/match_benchmark $(TARGET) tests/test_dict tests/test_movegen tests/test_eval tests/test_lattice_proof
