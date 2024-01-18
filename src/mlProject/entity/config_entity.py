from dataclasses import dataclass
from pathlib import Path


# ------------------------------------------ Data_Ingestion ------------------------------------


@dataclass(frozen = True)
class DataIngestionConfig:
    root_dir:Path
    source_URL : str
    local_data_file : Path
    unzip_dir : Path




# ------------------------------------------ Data_Validation ------------------------------------

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict



# ------------------------------------------ Data_Validation ------------------------------------
    
dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path


# ------------------------------------------ Data_Transformation ------------------------------------ 
    

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path