CXX      := g++
CXXFLAGS := -O3 -march=native -std=c++20 -flto -ffast-math -funroll-loops
CC       := $(CXX) $(CXXFLAGS)

BUILD_DIR := build
EXE      ?= $(BUILD_DIR)/bonsai
TARGET   := $(EXE)
SRCS      := bonsai.cpp

# build
default: $(TARGET)

$(TARGET): $(SRCS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# build and bench
bench: $(TARGET)
	./$(TARGET) bench

# just run
run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all bench run clean