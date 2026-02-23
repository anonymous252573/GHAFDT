import os
import shutil
import trimesh
import open3d as o3d
import numpy as np
from scipy.linalg import orthogonal_procrustes
import subprocess
import yaml
from scipy.spatial import cKDTree

# 1) ***** Center and ormalize the surfaces ******
def normalize_mesh(vertices):
    center = [sum(v[i] for v in vertices) / len(vertices) for i in range(3)]
    normalized_vertices = [[v[i] - center[i] for i in range(3)] for v in vertices]
    max_coord = max(max(abs(v[i]) for v in normalized_vertices) for i in range(3))
    normalized_vertices = [[v[i] / max_coord for i in range(3)] for v in normalized_vertices]
    return normalized_vertices

def read_off_file(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if lines[0].strip() == 'OFF':
            num_vertices, num_faces, _ = map(int, lines[1].strip().split())
            for line in lines[2:2 + num_vertices]:
                vertex = list(map(float, line.strip().split()))
                vertices.append(vertex)
            for line in lines[2 + num_vertices:]:
                face = list(map(int, line.strip().split()[1:]))
                faces.append(face)
    return vertices, faces

def write_off_file(file_path, vertices, faces):
    with open(file_path, 'w') as file:
        file.write('OFF\n')
        file.write(f'{len(vertices)} {len(faces)} 0\n')
        for vertex in vertices:
            file.write(' '.join(map(str, vertex)) + '\n')
        for face in faces:
            file.write(f'{len(face)} {" ".join(map(str, face))}\n')

# Define the function to process a single file
def process_file(input_file, output_folder):
    vertices, faces = read_off_file(input_file)
    normalized_vertices = normalize_mesh(vertices)
    output_file = os.path.join(output_folder, os.path.basename(input_file))
    write_off_file(output_file, normalized_vertices, faces)

def normalize(input_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)

    # Process each file in the input folder
    for file in files:
        if file.endswith(".off"):
            input_file = os.path.join(input_folder, file)
            process_file(input_file, output_folder)

    print("Normalization complete.")

# Folder containing your mesh files (same procedure for AD)
input_folder = r"C:\.....\NC_Offs"
output_folder = r"C:\.....\NC_Normalized_Offs"
normalize(input_folder)


# 2) ***** Align the surfaces ******
reference_mesh = r"C:\...\atlas.off"  # the atlas path

# Load the meshes
def load_mesh(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()
    return mesh

def save_off(file_path, mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    with open(file_path, 'w') as f:
        f.write("OFF\n")
        f.write(f"{vertices.shape[0]} {triangles.shape[0]} 0\n")
        for vertex in vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        for triangle in triangles:
            f.write(f"3 {triangle[0]} {triangle[1]} {triangle[2]}\n")

def procrustes_alignment(X, Y):
    A = np.dot(Y.T, X)
    U, S, Vt = np.linalg.svd(A)
    R = np.dot(U, Vt)
    Y_aligned = np.dot(Y, R)
    return Y_aligned

# R, _ = orthogonal_procrustes(Y, X)

def rigid_reg(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reference_pcd = load_mesh(reference_mesh)

    file_names = sorted([file_name for file_name in os.listdir(input_folder) if file_name.endswith('.off')])
    meshes = [load_mesh(os.path.join(input_folder, file_name)) for file_name in file_names]

    aligned_meshes_list = []
    for mesh in meshes:
        mesh_aligned = procrustes_alignment(np.asarray(reference_pcd.vertices), np.asarray(mesh.vertices))
        mesh.vertices = o3d.utility.Vector3dVector(mesh_aligned)
        aligned_meshes_list.append(mesh)

    for mesh, file_name in zip(aligned_meshes_list, file_names):
        output_path = os.path.join(output_folder, file_name)
        save_off(output_path, mesh)


# Folder containing your mesh files (same procedure for AD)
input_folder =  r"C:\....\NC_Normalized_Offs"
output_folder = r"C:\...\NC_Aligned_Offs"

rigid_reg(input_folder, output_folder)


# 3) ***** Surface Matching ******
# directory of your machine for DGCNN of Non-rigid registration  
config_file = r"C:\.....\DFR-main\registration2\config\scape_r.yaml"

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
    
scape_adni_folder = r"C:\......\DFR-main\registration2\data\scape_try"

temp_train_dir = os.path.join(scape_adni_folder, "shapes_train")
temp_test_dir = os.path.join(scape_adni_folder, "shapes_test")

# Define the command to run the registration
dgcnn_command = "python test.py"

def setup_directories():
    # Clear the contents of the shapes_train directory
    for filename in os.listdir(temp_train_dir):
        file_path = os.path.join(temp_train_dir, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    # Delete the cache files
    cache_files = [os.path.join(scape_adni_folder, 'cache_scape_dg_train.pt'), os.path.join(scape_adni_folder, 'cache_scape_dg_test.pt')]
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)

def run_registration(input_shape, input_filename):
    # Copy the input shape to the shapes_train directory with the actual filename
    dest_shape_path = os.path.join(temp_train_dir, input_filename)
    shutil.copy(input_shape, dest_shape_path)

    # Run the registration command
    try:
        subprocess.run(dgcnn_command, shell=True, check=True, cwd=r"C:\......\DFR-main\registration2")

    except subprocess.CalledProcessError as e:
        print(f"Registration command failed for {input_shape} with error: {e}")
        return
def run_reg(input_folder):
    # Loop through each input shape and deform it
    for filename in os.listdir(input_folder):
        if filename.endswith(".off"):
            input_shape = os.path.join(input_folder, filename)

            # Setup directories for each shape
            setup_directories()

            # Run registration for the current shape
            run_registration(input_shape, filename)


# Same procedure for AD
input_folder= r"C:\.....\CN_Aligned_Offs" #folder path of the aligned shapes
run_reg(input_folder)

# ******* extract the correspondence ********
atlas_file = r"C:\...\atlas.off"

deformed_folder = r"C:\.....\deformed_meshes"  #folder of deformed meshes from DGCNN
original_folder = r"C:\.....\aligned_meshes" #folder of aligned meshes

for filename in os.listdir(deformed_folder):
    if filename.endswith('_atlas.off'):
        new_filename = filename.replace('_atlas', '')
        old_file = os.path.join(deformed_folder, filename)
        new_file = os.path.join(deformed_folder, new_filename)
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} → {new_filename}")

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def extract_point_matches_ordered(atlas_file, deformed_file, original_file):
    # Load atlas, deformed, and original meshes
    atlas_mesh = trimesh.load_mesh(atlas_file, process=False) 
    deformed_mesh = trimesh.load_mesh(deformed_file)
    original_mesh = trimesh.load_mesh(original_file)

    # Extract vertices from meshes
    atlas_points = np.array(atlas_mesh.vertices)
    deformed_points = np.array(deformed_mesh.vertices)
    original_points = np.array(original_mesh.vertices)
    
    deformed_tree = cKDTree(deformed_points)
    _, nearest_indices = deformed_tree.query(atlas_points)
    corresponding_original_points = original_points[nearest_indices]

    point_matches = [(atlas_points[i], corresponding_original_points[i]) for i in range(len(atlas_points))]
    return point_matches

def extracting_matches(atlas_file, deformed_folder, original_folder, matches_folder):
    deformed_files = sorted(os.listdir(deformed_folder), key=natural_sort_key)
    original_files = sorted(os.listdir(original_folder), key=natural_sort_key)

    assert len(deformed_files) == len(original_files), "The number of deformed shapes and original shapes must match."

    for i in range(len(deformed_files)):
        deformed_file = os.path.join(deformed_folder, deformed_files[i])
        original_file = os.path.join(original_folder, original_files[i])

        point_matches = extract_point_matches_ordered(atlas_file, deformed_file, original_file)

        # Save point matches using absolute path to the matches folder
        matches_file = os.path.join(matches_folder, f"matched_{os.path.splitext(deformed_files[i])[0]}.txt")
        with open(matches_file, 'w') as f:
            for match in point_matches:
                atlas_point_str = ' '.join(map(str, match[0]))
                original_point_str = ' '.join(map(str, match[1]))
                f.write(f"{atlas_point_str} -> {original_point_str}\n")

        print(f"Extracted point matches for {deformed_files[i]} and saved to {matches_file}")


matches_folder = r"C:\.....\matched_meshes"   # output folder here
extracting_matches(atlas_file, deformed_folder, original_folder, matches_folder)
