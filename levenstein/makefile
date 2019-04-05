#############################################################################

HW_NAME = $(SLURM_JOB_PARTITION)

#############################################################################

# defaults:
HW_GCCOPTS = -mavx2 -D USE_AVX
HW_RUNOPTS = avx

ifeq "$(HW_NAME)" "small-lp"
HW_GCCOPTS = -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl -D USE_AVX512
HW_RUNOPTS = avx512
endif

ifeq "$(HW_NAME)" "big-lp"
HW_GCCOPTS = -msse4.2
HW_RUNOPTS = sse
endif

TEST_VERSION = true

#############################################################################

RELEASE_CPP_NAMES = \
	du4levenstein \
	exp4main

RELEASE_HEADER_FILES = \
	du4levenstein.hpp \
	task4.hpp \
	testbed.hpp

TEST_CPP_NAMES = \
	du4levenstein \
	my_main

TEST_HEDER_FILES = \
	du4levenstein.hpp \
	functional_tester.hpp \
	levenstein_tester_sse.hpp \
	levenstein_tester_avx512.hpp \
	dummy_levenstein.hpp

ifeq "$(TEST_VERSION)" "true"
	CPP_NAMES = $(TEST_CPP_NAMES)
	HEADER_FILES = $(TEST_HEADER_FILES)
else
	CPP_NAME = $(RELEASE_CPP_NAMES)
	HEADER_FILES = $(RELEASE_HEADER_FILES)
endif

TEMP_FOLDER = tmp.$(HW_NAME)

TARGET_NAME = du4

#############################################################################

TEMP_FOLDER_FILE = $(TEMP_FOLDER)/here.iam

LINKED_NAMES = $(CPP_NAMES)

OBJ_NAMES = $(CPP_NAMES)

LINKED_OBJS = $(LINKED_NAMES:%=$(TEMP_FOLDER)/%.o)

OBJ_FILES = $(OBJ_NAMES:%=$(TEMP_FOLDER)/%.o)

TARGET_FILE = $(TEMP_FOLDER)/$(TARGET_NAME)

OUT_FILE = $(TARGET_NAME).out.$(HW_NAME).txt

BYPRODUCT_FILES = 

TEMPORARY_FILES = $(BYPRODUCT_FILES) $(OBJ_FILES) $(TARGET_FILE) $(OUT_FILE) 

INCLUDED_HEADER_FILES = $(HEADER_FILES)

INCLUDE_OPTS = \
	-I.

ifeq "$(TEST_VERSION)" "true"
	COMPILE_COMMAND = $(CXX) -c -ggdb -std=c++17 $(HW_GCCOPTS) $(INCLUDE_OPTS) -o$@ $<
else
	COMPILE_COMMAND = $(CXX) -c -ggdb -O3 -std=c++17 $(HW_GCCOPTS) $(INCLUDE_OPTS) -o$@ $<
endif

LINK_COMMAND = $(CXX) -o$@ $^ 

RUN_COMMAND = $< $(HW_RUNOPTS) | tee $@

#############################################################################

VPATH = \
	$(TEMP_FOLDER)

.PHONY: all
all: $(TEMP_FOLDER_FILE) $(TARGET_NAME)

.PHONY: info
info:
	@echo HW_NAME=$(HW_NAME)
	@echo HW_GCCOPTS=$(HW_GCCOPTS)
	@echo HW_RUNOPTS=$(HW_RUNOPTS)

.PHONY: run
run: $(TEMP_FOLDER_FILE) $(OUT_FILE)

.PHONY: $(TARGET_NAME)
$(TARGET_NAME): $(TARGET_FILE)

.PHONY: clean
clean:
	rm $(TEMPORARY_FILES)

.SECONDARY: $(TEMPORARY_FILES)

$(TEMP_FOLDER_FILE):
	mkdir -p $(TEMP_FOLDER)
	touch $@

$(OUT_FILE): $(TARGET_FILE)
	$(RUN_COMMAND)

$(TARGET_FILE): $(LINKED_OBJS)
	$(LINK_COMMAND)

$(TEMP_FOLDER)/%.o: %.cpp $(INCLUDED_HEADER_FILES)
	$(COMPILE_COMMAND)

##################################################################################
