CXX      := g++
CXXFLAGS := -O3 -march=native -std=c++20 -flto -ffast-math -funroll-loops

BUILD_DIR := build
TARGET    := $(BUILD_DIR)/bonsai
SRCS      := bonsai.cpp

# build
all: $(TARGET)

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