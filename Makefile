CC=g++
INCLUDES=-Iinclude
BUILD_FOLDER=build
TARGETS=test_ops
TARGET_PATH=$(BUILD_FOLDER)/$(TARGETS)
SRC_DIR=src
TEST_DIR=tests

SRCS=$(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(TEST_DIR)/*.cpp)
OBJS=$(patsubst %.cpp, $(BUILD_FOLDER)/%.o, $(notdir $(SRCS)))

.PHONY: clean check

vpath %.cpp $(SRC_DIR) $(TEST_DIR)

$(TARGET_PATH) : $(OBJS)
	$(CC) $(INCLUDES) $^ -o $@

$(BUILD_FOLDER)/%.o : %.cpp | $(BUILD_FOLDER)
	$(CC) $(INCLUDES) -c $< -o $@

$(BUILD_FOLDER):
	mkdir -p $@

check : $(TARGET_PATH)
	./$(TARGET_PATH)

clean :
	rm -rf $(BUILD_FOLDER)