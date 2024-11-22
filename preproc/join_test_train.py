import os
import shutil
import sys
import json
from typing import Any
from alive_progress import alive_bar

def move_files(source_dir:str, dest_dir:str, bar:Any) -> None:
    '''
        Moves all files in the source directory to the destination directory.
        Args:
            source_dir (str): source directory.
            dest_dir (str): destination directory.
            bar (Any): object to update th progress bar.
    '''
    files_to_move = os.listdir(source_dir)
    for file_name in files_to_move:
        source_path = os.path.join(source_dir, file_name)
        shutil.move(source_path, dest_dir)
        bar()

if __name__ == '__main__':
    '''
        Splits the dataset into train and test set.

        Parameters:
        -dp data path
    '''
    i = 1
    data_path = None
    while i < len(sys.argv):
        if sys.argv[i] == "-dp":
            if data_path != None: raise ValueError("Repeated input for data path")
            data_path = sys.argv[i+1]; i+= 2
        else:
            i+=1
    if data_path is None: raise TypeError("Missing value for data path")

    map_path = os.path.join(data_path, "cms2ogs.json")

    with open(map_path, 'r') as f:
        cms2ogs_map = json.load(f)

    var_list = list(cms2ogs_map.keys())

    folder_3D = "original"
    folder_surface = "surface"

    cms_path_3D = os.path.join(data_path, folder_3D, "nc_iCMS")
    ogs_path_3D = os.path.join(data_path, folder_3D, "nc_OGS")
    cms_test_path_3D = os.path.join(data_path, folder_3D, "nc_iCMS_test")
    ogs_test_path_3D = os.path.join(data_path, folder_3D, "nc_OGS_test")
    cms_path_surface = os.path.join(data_path, folder_surface, "nc_iCMS")
    ogs_path_surface = os.path.join(data_path, folder_surface, "nc_OGS")
    cms_test_path_surface = os.path.join(data_path, folder_surface, "nc_iCMS_test")
    ogs_test_path_surface = os.path.join(data_path, folder_surface, "nc_OGS_test")

    print("[joining_test_train] Starting execution")
    with alive_bar(0, title=f"Moving files...") as bar:
        for var in var_list:
            # Joining 3D data
            source_dir_cms = os.path.join(cms_test_path_3D, var)
            source_dir_ogs = os.path.join(ogs_test_path_3D, cms2ogs_map[var])
            dest_dir_cms = os.path.join(cms_path_3D, var)
            dest_dir_ogs = os.path.join(ogs_path_3D, cms2ogs_map[var])
            move_files(source_dir_cms, dest_dir_cms, bar)
            move_files(source_dir_ogs, dest_dir_ogs, bar)
            # Joining surface data
            source_dir_cms = os.path.join(cms_test_path_surface, var)
            source_dir_ogs = os.path.join(ogs_test_path_surface, cms2ogs_map[var])
            dest_dir_cms = os.path.join(cms_path_surface, var)
            dest_dir_ogs = os.path.join(ogs_path_surface, cms2ogs_map[var])
            move_files(source_dir_cms, dest_dir_cms, bar)
            move_files(source_dir_ogs, dest_dir_ogs, bar)
        # Joining rivers data
        source_dir_riv = os.path.join(data_path, "rivers", "vector_test")
        dest_dir_riv = os.path.join(data_path, "rivers", "vector")
        move_files(source_dir_riv, dest_dir_riv, bar)
    print("[joining_test_train] Ending execution")
