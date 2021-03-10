import sys
from optparse import OptionParser, OptionGroup

usage ="""
<script>.py
Options:
  -h, --help            show this help message and exit
  -d /data, --data-dir=/data
                        root directory of your data
  -m lh,rh, --hemisphere=lh,rh
                        hemisphere list
  -n 50, --num-tests=50
                        Number of tests
  -a DKTatlas40, --atlas=DKTatlas40
                        Atlas
"""

parser = OptionParser(usage=usage)

parser.add_option("-d", "--data-dir",  action="store", type="string", dest="data_dir", help="root directory of your data", metavar="/data/MASTROGIOVANNI", default="/data/MASTROGIOVANNI")
parser.add_option("-m", "--hemisphere",  action="store", type="string", dest="hemisphere", help="hemisphere list", metavar="lh,rh", default="lh,rh")
parser.add_option("-n", "--num-tests",  action="store", type="int", dest="num_tests", help="Number of tests", metavar="50", default="50")
parser.add_option("-a", "--atlas",  action="store", type="string", dest="atlas", help="Atlas", metavar="DKTatlas40", default="DKTatlas40")
parser.add_option("-c", "--computation-id",  action="store", type="string", dest="computation_id", help="Computation ID: this is the prefix for your computation", metavar="")
parser.add_option("-s", "--skip-tests",  type="int", dest="skip_tests", help="Skip this number of tests", default="0")
parser.add_option("-t", "--subject-type",  action="store", type="string", dest="subject_type", help="Subject type", metavar="controllo", default="controllo")
parser.add_option("-u", "--subject-id",  action="store", type="string", dest="subject_id", help="Subject ID", metavar="0040013", default="0040013")

options, args = parser.parse_args()

if len(args) > 0:
    sys.stderr.write("Too many arguments!  Try --help.")
    raise SystemExit()

if '-h' in sys.argv:
    parser.print_help()
    raise SystemExit()

data_dir = options.data_dir
hemis = options.hemisphere.split(",")
num_tests = options.num_tests
atlas = options.atlas
computation_id = options.computation_id
subject_type = options.subject_type
subject_id = options.subject_id

skip_tests = options.skip_tests

# Parcels directory
DIR_PARCELS = f"{computation_id}/random-parcels"

# fsaverage directory
fsaverage_dir = f"{data_dir}/fsaverage/"

# Indexes
# indexes = ["origin"]
indexes = list(map(str, range(num_tests)))
indexes.insert(0, "origin")

if computation_id == None:
    print("------------------------------------------------------")
    print(f"The system cannot run without computation ID")
    print("------------------------------------------------------")
    sys.exit(0)

print("------------------------------------------------------")
print (f"The system is running: [ ID: {computation_id} ]")
print(f"data_dir:      {data_dir}")
print(f"hemis:         {hemis}")
print(f"num_tests:     {num_tests}")
print(f"indexes:       {indexes}")
print("------------------------------------------------------")

import numpy as np
import os
import nibabel.freesurfer.io as fio

def get_size(vertices, num_parcels):
    return np.array([np.sum(np.array(vertices) == i) for i in range(num_parcels)])

"""
Return the average percentual change in size for parcel
"""
def get_absolute_differences(vertices_before, vertices_after, num_parcels):
    counts_before = get_size(vertices_before, num_parcels)
    counts_after = get_size(vertices_after, num_parcels)
    return np.abs(counts_after - counts_before) / counts_before * 100.0

"""
Perform the parcelization based on this method:
"""
class Parcelization(object):

    def __init__(self, fsaverage_dir, hemisphere, atlas):

        # Load patient labels for vertex based on <atlas> parcels (MACRO)
        (self.vertices, self.colortable, self.labels) = fio.read_annot(
            f"{fsaverage_dir}/label/{hemisphere}.{atlas}.annot")

        # Load geometry of the brain
        (self.coords, self.faces) = fio.read_geometry(f"{fsaverage_dir}/surf/{hemisphere}.sphere.reg")

        # an array of list: at position i-th there is the list of faces that the vertex i touches
        self.vertex_to_faces = np.empty((len(self.vertices),), dtype=object)
        for i in range(len(self.vertex_to_faces)):
            self.vertex_to_faces[i] = []
        # For each face, add the face i to the vertex i-th list of faces
        for i, f in enumerate(self.faces):
            for j in range(3):
                self.vertex_to_faces[f[j]].append(i)

    def get_vertices(self):
        return self.vertices

    def get_faces(self):
        return self.faces

    def __get_boundary_faces_to_parcel(self, parcel):

        # Create an array with faces as row and in column
        # the parcel where the vertex belongs to:
        # vf[i] = [a, b, c]
        # means that face i has:
        # - vertex 1 on parcel a
        # - vertex 2 on parcel b
        # - vertex 3 on parcel c
        vf = self.vertices[self.faces]

        # Identify faces that belong to different macros

        # This is an array that is True in the i-th position
        # if the i-th face has some vertices in a parcel but
        # others in other parcel. None of the vertex must be in
        # the "unknown" parcel (0).
        faces_of_edge_micro = np.logical_and(
            # Face has vertex in different parcels
            np.logical_or(
                vf[:, 0] != vf[:, 1],
                vf[:, 1] != vf[:, 2],
                vf[:, 2] != vf[:, 0]
            ),
            # One of three vertex must be in this macro
            np.logical_or(
                vf[:, 0] == parcel,
                vf[:, 1] == parcel,
                vf[:, 2] == parcel
            )
        )

        # index of faces of edge micro
        [index_faces_of_edge_micro] = np.where(faces_of_edge_micro == True)
        return index_faces_of_edge_micro

    """
    Return first vertex on the boundary of a parcel
    """

    def __get_first_vertex_outside_parcel(self, parcel):
        pf = self.__get_boundary_faces_to_parcel(parcel)
        first_pf = self.faces[pf[0]]
        
        # print("Parcel:", parcel)
        # print("Face:", first_pf)
        # print("Parcels:", self.vertices[first_pf])

        for i in range(3):
            if self.vertices[first_pf[i]] != parcel:
                return first_pf[i]

    """
    Return the list of faces that are on multiple parcels
    """

    def get_peripheral_faces(self):

        # Create an array with faces as row and in column
        # the parcel where the vertex belongs to:
        # vf[i] = [a, b, c]
        # means that face i has:
        # - vertex 1 on parcel a
        # - vertex 2 on parcel b
        # - vertex 3 on parcel c
        vf = self.vertices[self.faces]

        # Identify faces that belong to different macros

        # This is an array that is True in the i-th position
        # if the i-th face has some vertices in a parcel but
        # others in other parcel. None of the vertex must be in
        # the "unknown" parcel (0).
        faces_of_edge_micro = np.logical_and(
            # Some of the parcel is different from another in the trio
            np.logical_or(
                vf[:, 0] != vf[:, 1],
                vf[:, 1] != vf[:, 2],
                vf[:, 2] != vf[:, 0]
            ),
            # Excludes macro "unknown" (0)from the micro parcel you can absorb
            np.logical_and(
                vf[:, 0] != 0,
                vf[:, 1] != 0,
                vf[:, 2] != 0
            )
        )

        # index of faces of edge micro
        [index_faces_of_edge_micro] = np.where(faces_of_edge_micro == True)
        return index_faces_of_edge_micro

    """
    Return the list of faces that has one on the "in_vertices" inside
    """

    def faces_from_vertex(self, in_vertices):
        return np.unique([face for face in self.vertex_to_faces[in_vertices]])

        # for v in in_vertices:
        #    for face in self.vertex_to_faces[v]:
        #        out_faces.append(face)
        # return np.unique(out_faces)

    """
    return list of vertexes given a set of faces
    the first parameter map faces to vertex
    """

    def vertexes_from_faces(self, in_faces):
        out_vertexes = []
        for f in in_faces:
            for v in self.faces[f]:
                out_vertexes.append(v)
        return np.unique(out_vertexes)

    """
    Dump size of each parcel
    """

    def __dump_parcels_size(self):
        for i in range(len(self.labels)):
            print(i, ":", np.sum(self.vertices == i))

    # grow_up_parcels choose at random "num_parcels" parcels and increase each one
    # of a number of vertices such that the sum of vertices changed reach a the "threshold"
    # specified over total number of vertices.
    # Parcel 0 must be skipped.
    # Parcel with size 0 must be skipped
    def grow_up_parcels(self, num_parcels, threshold):

        # Size of each parcel
        parcel_size = get_size(self.vertices, len(self.labels))

        # Number of parcel excluding the one with 0 size and parcel 0
        tot_parcels = np.sum(np.array(parcel_size[1:]) > 0)

        if num_parcels > tot_parcels:
            raise Exception(f"Cannot increase {num_parcels}: total is {tot_parcels}")

        # Total number of vertices excluding the parcel 0
        total_number_of_vertices = np.sum(parcel_size[1:])

        # Permutations of parcels (0 excluded)
        permutations = np.random.permutation(len(self.labels))

        parcels = np.array(range(len(self.labels)))
        parcels = parcels[permutations[np.where(parcels[permutations] > 0)]]
        print("Parcels to check:", parcels)

        index_parcel = 0
        parcels_to_increase = []
        
        while len(parcels_to_increase) < num_parcels:

            parcel_to_increase = parcels[index_parcel]
            if parcel_size[parcel_to_increase] == 0:
                # print("Skip parcel", parcel_to_increase, "because empty")
                index_parcel += 1
                continue

            # print("Chosen parcel", parcel_to_increase, "Increase it!")
            index_parcel += 1
            parcels_to_increase.append(parcel_to_increase)

        # print("Parcels:", parcels_to_increase)

        total_size_of_parcels_to_increase = np.sum(parcel_size[parcels_to_increase])

        # print("Total size:", total_size_of_parcels_to_increase)
        # print("Partial sizes:", parcel_size[parcels_to_increase])
        # print("Sizes:", parcel_size)

        vertex_to_move = (threshold * total_number_of_vertices * parcel_size[parcels_to_increase]) / (
                100.0 * total_size_of_parcels_to_increase)

        # print("Vertex to move:", vertex_to_move)

        # Save initial status and dump parcel size
        before = np.array(self.vertices)
        # self.__dump_parcels_size()

        for i in range(num_parcels):

            parcel_to_increase = parcels_to_increase[i]
            num_vertex_to_add = vertex_to_move[i]

            print("Parcel to increase: ", parcel_to_increase, " vertex to move:", num_vertex_to_add)

            # Find first vertex outside parcel
            first_vertex = self.__get_first_vertex_outside_parcel(parcel_to_increase)
            print("First vertex: ", first_vertex)
            added_vertices = np.array([first_vertex])

            # print("Want to add", num_vertex_to_add, "vertices")

            # print("Intial external:", added_vertices)

            old_count = 0
            while len(added_vertices) < num_vertex_to_add:
                old_count = len(added_vertices)
                _faces = self.faces_from_vertex(added_vertices)
                _vertices = self.vertexes_from_faces(_faces)
                added_vertices = _vertices[np.where(self.vertices[_vertices] != parcel_to_increase)]
                if old_count == len(added_vertices):
                    # print("Parcel did'nt grow")
                    break

            # print("Will add", len(added_vertices), "vertices")

            self.vertices[added_vertices] = parcel_to_increase

        diff = get_absolute_differences(before, self.vertices, len(self.labels))
        print("Differences:", np.nanmean(diff))
        # self.__dump_parcels_size()

print("Creating parcels in data/parcels...")

# Create destination directory
print(f"Creating: {data_dir}/recon_all/{subject_type}/{subject_id}/{DIR_PARCELS}")
os.makedirs(f"{data_dir}/recon_all/{subject_type}/{subject_id}/{DIR_PARCELS}", exist_ok=True)

# we want to generate 10 parcels for the given hemisfere
for index in range(num_tests):

    if index < skip_tests:
        next

    for hemi in hemis:
    
        done = False
        while not done:
            done = True
            try:
                p = Parcelization(fsaverage_dir, hemi, atlas)

                p.grow_up_parcels(10, 3.0) # aparc 23, DKTatlas40 10

                print("I'm writing the annotation file")

                fio.write_annot(f"{data_dir}/recon_all/{subject_type}/{subject_id}/{DIR_PARCELS}/{index}.{hemi}.annot", p.vertices, p.colortable, p.labels)

                (v, c, l) = fio.read_annot(f"{data_dir}/recon_all/{subject_type}/{subject_id}/{DIR_PARCELS}/{index}.{hemi}.annot")

                print("Saved:", f"{data_dir}/recon_all/{subject_type}/{subject_id}/{DIR_PARCELS}/{index}.{hemi}.annot", "with:", str(len(l)), "Labels")

            except:
                done = False

