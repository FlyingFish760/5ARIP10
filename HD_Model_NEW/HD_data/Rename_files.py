import os

new_prefix = 'F1_dSpace_'  # Replace with the string you want to add as a prefix
sensor_folder = 'dSpace' # Choose 'Microphone', 'Vibration' 'dSPace'
folder_path = os.path.join(os.getcwd(), 'HD_Model_NEW', 'HD_data','Faulty',sensor_folder) 

# Code the add a prefix to the file
# Get all the file names in the folder
file_names = os.listdir(folder_path)

# Iterate over each file name and rename it
for file_name in file_names:
    # Construct the new file name with the added prefix
    new_file_name = new_prefix + file_name

    # Get the current file path and the new file path
    current_path = os.path.join(folder_path, file_name)
    new_path = os.path.join(folder_path, new_file_name)

    # Rename the file
    os.rename(current_path, new_path)


# Code used to remove parts in the file name
# portion_to_remove = new_prefix #'remove_'  # Replace with the portion you want to remove

# Get all the file names in the folder
# file_names = os.listdir(folder_path)

# Iterate over each file name and rename it
# for file_name in file_names:
#     # Check if the portion to remove exists in the file name
#     if portion_to_remove in file_name:
#         # Construct the new file name without the specified portion
#         new_file_name = file_name.replace(portion_to_remove, '')
        
#         # Get the current file path and the new file path
#         current_path = os.path.join(folder_path, file_name)
#         new_path = os.path.join(folder_path, new_file_name)

#         # Rename the file
#         os.rename(current_path, new_path)