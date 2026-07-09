CC=g++
INCLUDES=-Iinclude
BUILD_FOLDER=build
TARGETS=generate
TARGET_PATH=$(BUILD_FOLDER)/$(TARGETS)
SRC_DIR=src

SRCS=$(wildcard $(SRC_DIR)/*.cpp)
TESTS=$(wildcard tests/*.cpp)
OBJS=$(patsubst %.cpp, $(BUILD_FOLDER)/%.o, $(notdir $(SRCS)))

.PHONY: clean check

vpath %.cpp $(SRC_DIR)

$(TARGET_PATH) : $(OBJS)
	$(CC) $(INCLUDES) $^ -o $@

$(BUILD_FOLDER)/%.o : %.cpp | $(BUILD_FOLDER)
	$(CC) $(INCLUDES) -c $< -o $@

$(BUILD_FOLDER):
	mkdir -p $@

$(BUILD_FOLDER)/check.o : tests/test_ops.cpp | $(BUILD_FOLDER)
	$(CC) $(INCLUDES) -c $< -o $@

$(BUILD_FOLDER)/test_encoding.o : tests/test_encoding.cpp | $(BUILD_FOLDER)
	$(CC) $(INCLUDES) -c $< -o $@

build/check : build/check.o build/ops.o 
	$(CC) $(INCLUDES) -o $@ $^

build/test_encoding : build/llama.o build/test_encoding.o
	$(CC) $(INCLUDES) -o $@ $^

check : build/check build/test_encoding
	./build/check
	./build/test_encoding

clean :
	rm -rf $(BUILD_FOLDER)