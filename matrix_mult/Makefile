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

#############################################################################

CPP_NAMES = \
	du5matrix \
	du5main

HEADER_FILES = \
	du5matrix.hpp \
	task5.hpp \
	testbed.hpp

TEMP_FOLDER = tmp.$(HW_NAME)

TARGET_NAME = du5

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

COMPILE_COMMAND = $(CXX) -c -ggdb -O3 -std=c++17 $(HW_GCCOPTS) $(INCLUDE_OPTS) -o$@ $<

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

FILES = \
	Makefile \
	du5main.cpp \
	du5matrix.cpp \
	du5matrix.hpp \
	task5.hpp \
	testbed.hpp \
	du5.err.gold.txt

SSH_CMD = ssh parlab
PROJECT_DIR = /mnt/home/s_bruchpa/matrix_mult_vyvyvys

PARTITION = small-lp

SRUN_CMD = srun -p $(PARTITION)

send_files:
	@echo "Sending files..."
	scp -r -P 42222 $(FILES) "s_bruchpa@parlab.ms.mff.cuni.cz:$(PROJECT_DIR)"

compile: send_files
	@echo "Compiling..."
	$(SSH_CMD) "cd $(PROJECT_DIR); srun -p $(PARTITION) $(MAKE)"

