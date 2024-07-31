define ANNOUNCE_BODY

Makefile to build the UGV Main Application Unit Tests
-----------------------------------------------------

Prerequisits:
-------------

Debian/Ubuntu:
sudo apt-get install libeigen3-dev

openSUSE:
sudo zypper install eigen3-devel
endef

# CC=gcc
# CXX=g++

ROOTDIR = ../..
BUILD_DIR = build
TARGET = test


CXXFLAGS= -Wall -Wextra -g -ggdb -I${ROOTDIR}/include -I$(ROOTDIR)/src -I$(ROOTDIR)/src/ips -I/usr/include/eigen3
CFLAGS  = -Wall -Wextra -g -ggdb -I${ROOTDIR}/include -I$(ROOTDIR)/src -I$(ROOTDIR)/src/ips -I/usr/include/eigen3

LDFLAGS=-g

# Generate dependency files, these can be included by the makefile (see -include in last line)
DEPENDENCYFLAGS = -MMD -MP -MF"$(@:%.o=%.d)"

CXX_SOURCES = \
	main.cpp

C_SOURCES =


# automatically generate a list of all project objects
C_OBJECTS = $(addprefix $(BUILD_DIR)/,$(notdir $(patsubst %.c, %.o, $(C_SOURCES))))
vpath %.c .:$(sort $(dir $(C_SOURCES)))

CXX_OBJECTS  = $(addprefix $(BUILD_DIR)/,$(notdir $(patsubst %.cpp, %.o, $(CXX_SOURCES))))
vpath %.cpp .:$(sort $(dir $(CXX_SOURCES)))

.PHONY: all
all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR)/$(TARGET): $(CXX_OBJECTS) $(C_OBJECTS) | $(BUILD_DIR)
	@echo "--- linking $@"
	@$(CXX) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o:%.cpp | $(BUILD_DIR)
	@echo "--- compiling $@ from $<"
	@$(CXX) -c $< -o $@ $(CXXFLAGS) $(DEPENDENCYFLAGS)

$(BUILD_DIR)/%.o:%.c | $(BUILD_DIR)
	@echo "--- compiling $@ from $<"
	@$(CC) -c $< -o $@ $(CFLAGS) $(DEPENDENCYFLAGS)

$(BUILD_DIR):
	@echo "--- create build directory $@"
	@mkdir $@

clean:
	@echo "--- remove build directory $(BUILD_DIR)"
	@rm -rf $(BUILD_DIR)

export ANNOUNCE_BODY
.PHONY: help
help:
	@echo "$$ANNOUNCE_BODY"

# include dependency files generated by the compiler
-include $(wildcard $(BUILD_DIR)/*.d)