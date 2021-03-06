#############################################################################

HW_NAME = $(SLURM_JOB_PARTITION)

#############################################################################

# defaults:
ifeq "$(HW_NAME)" ""
HW_NAME = default
endif
HW_GCCOPTS = -msse4.2 -mavx2 -D USE_AVX
HW_RUNOPTS = sse avx

ifeq "$(HW_NAME)" "small-lp"
HW_GCCOPTS = -msse4.2 -mavx2 -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl -D USE_AVX -D USE_AVX512
HW_RUNOPTS = sse avx avx512
endif

ifeq "$(HW_NAME)" "big-lp"
HW_GCCOPTS = -msse4.2
HW_RUNOPTS = sse
endif

ifeq "$(HW_NAME)" "phi-lp"
HW_GCCOPTS = -msse4.2 -mavx2 -mavx512f -mavx512pf -mavx512er -mavx512cd -D USE_AVX -D USE_AVX512
HW_RUNOPTS = sse avx avx512
endif

TEST_VERSION = false

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
	CPP_NAMES = $(RELEASE_CPP_NAMES)
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

OUT_FILE_PREF = $(TARGET_NAME).out.$(HW_NAME).
OUT_FILE_SUFF = .txt
ERR_FILE_PREF = $(TARGET_NAME).err.$(HW_NAME).
ERR_FILE_SUFF = .txt

OUT_FILES = $(HW_RUNOPTS:%=$(OUT_FILE_PREF)%$(OUT_FILE_SUFF))

BYPRODUCT_FILES = 

TEMPORARY_FILES = $(BYPRODUCT_FILES) $(OBJ_FILES) $(TARGET_FILE) $(OUT_FILES) 

INCLUDED_HEADER_FILES = $(HEADER_FILES)

INCLUDE_OPTS = \
	-I.

ifeq "$(TEST_VERSION)" "true"
	COMPILE_COMMAND = $(CXX) -c -ggdb -O3 -std=c++17 $(HW_GCCOPTS) $(INCLUDE_OPTS) -o$@ $<
else
	COMPILE_COMMAND = $(CXX) -c -ggdb -O3 -std=c++17 $(HW_GCCOPTS) $(INCLUDE_OPTS) -o$@ $<
endif

LINK_COMMAND = $(CXX) -o$@ $^ 

RUN_COMMAND = $< $* 2> $(ERR_FILE_PREF)$*$(ERR_FILE_SUFF) | tee $@

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

.PHONY: rerun
rerun: cleanout run

.PHONY: run
run: $(TEMP_FOLDER_FILE) $(OUT_FILES)

.PHONY: $(TARGET_NAME)
$(TARGET_NAME): $(TARGET_FILE)

.PHONY: clean
clean:
	-rm $(TEMPORARY_FILES)

.PHONY: cleanout
cleanout:
	-rm $(OUT_FILES)

.SECONDARY: $(TEMPORARY_FILES)

$(TEMP_FOLDER_FILE):
	mkdir -p $(TEMP_FOLDER)
	touch $@

$(OUT_FILE_PREF)%$(OUT_FILE_SUFF): $(TARGET_FILE)
	$(RUN_COMMAND)

$(TARGET_FILE): $(LINKED_OBJS)
	$(LINK_COMMAND)

$(TEMP_FOLDER)/%.o: %.cpp $(INCLUDED_HEADER_FILES)
	$(COMPILE_COMMAND)

##################################################################################
