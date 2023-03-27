#-*- coding: UTF-8 -*-

"""
Some fancy helper functions.
"""

import os
import ctypes

from distutils.sysconfig import get_python_lib
import re
import sys

try: import numpy
except ImportError: numpy = None

import logging;logger = logging.getLogger("pyassimp")

from .errors import AssimpError

additional_dirs, ext_whitelist = [],[]

def vec2tuple(x):
    """ Converts a VECTOR3D to a Tuple """
    return (x.x, x.y, x.z)

def transform(vector3, matrix4x4):
    """ Apply a transformation matrix on a 3D vector.

    :param vector3: array with 3 elements
    :param matrix4x4: 4x4 matrix
    """
    if numpy:
        return numpy.dot(matrix4x4, numpy.append(vector3, 1.))
    else:
        m0,m1,m2,m3 = matrix4x4; x,y,z = vector3
        return [
            m0[0]*x + m0[1]*y + m0[2]*z + m0[3],
            m1[0]*x + m1[1]*y + m1[2]*z + m1[3],
            m2[0]*x + m2[1]*y + m2[2]*z + m2[3],
            m3[0]*x + m3[1]*y + m3[2]*z + m3[3]
            ]

def _inv(matrix4x4):
    m0,m1,m2,m3 = matrix4x4

    det  =  m0[3]*m1[2]*m2[1]*m3[0] - m0[2]*m1[3]*m2[1]*m3[0] - \
            m0[3]*m1[1]*m2[2]*m3[0] + m0[1]*m1[3]*m2[2]*m3[0] + \
            m0[2]*m1[1]*m2[3]*m3[0] - m0[1]*m1[2]*m2[3]*m3[0] - \
            m0[3]*m1[2]*m2[0]*m3[1] + m0[2]*m1[3]*m2[0]*m3[1] + \
            m0[3]*m1[0]*m2[2]*m3[1] - m0[0]*m1[3]*m2[2]*m3[1] - \
            m0[2]*m1[0]*m2[3]*m3[1] + m0[0]*m1[2]*m2[3]*m3[1] + \
            m0[3]*m1[1]*m2[0]*m3[2] - m0[1]*m1[3]*m2[0]*m3[2] - \
            m0[3]*m1[0]*m2[1]*m3[2] + m0[0]*m1[3]*m2[1]*m3[2] + \
            m0[1]*m1[0]*m2[3]*m3[2] - m0[0]*m1[1]*m2[3]*m3[2] - \
            m0[2]*m1[1]*m2[0]*m3[3] + m0[1]*m1[2]*m2[0]*m3[3] + \
            m0[2]*m1[0]*m2[1]*m3[3] - m0[0]*m1[2]*m2[1]*m3[3] - \
            m0[1]*m1[0]*m2[2]*m3[3] + m0[0]*m1[1]*m2[2]*m3[3]

    return[[( m1[2]*m2[3]*m3[1] - m1[3]*m2[2]*m3[1] + m1[3]*m2[1]*m3[2] - m1[1]*m2[3]*m3[2] - m1[2]*m2[1]*m3[3] + m1[1]*m2[2]*m3[3]) /det,
            ( m0[3]*m2[2]*m3[1] - m0[2]*m2[3]*m3[1] - m0[3]*m2[1]*m3[2] + m0[1]*m2[3]*m3[2] + m0[2]*m2[1]*m3[3] - m0[1]*m2[2]*m3[3]) /det,
            ( m0[2]*m1[3]*m3[1] - m0[3]*m1[2]*m3[1] + m0[3]*m1[1]*m3[2] - m0[1]*m1[3]*m3[2] - m0[2]*m1[1]*m3[3] + m0[1]*m1[2]*m3[3]) /det,
            ( m0[3]*m1[2]*m2[1] - m0[2]*m1[3]*m2[1] - m0[3]*m1[1]*m2[2] + m0[1]*m1[3]*m2[2] + m0[2]*m1[1]*m2[3] - m0[1]*m1[2]*m2[3]) /det],
           [( m1[3]*m2[2]*m3[0] - m1[2]*m2[3]*m3[0] - m1[3]*m2[0]*m3[2] + m1[0]*m2[3]*m3[2] + m1[2]*m2[0]*m3[3] - m1[0]*m2[2]*m3[3]) /det,
            ( m0[2]*m2[3]*m3[0] - m0[3]*m2[2]*m3[0] + m0[3]*m2[0]*m3[2] - m0[0]*m2[3]*m3[2] - m0[2]*m2[0]*m3[3] + m0[0]*m2[2]*m3[3]) /det,
            ( m0[3]*m1[2]*m3[0] - m0[2]*m1[3]*m3[0] - m0[3]*m1[0]*m3[2] + m0[0]*m1[3]*m3[2] + m0[2]*m1[0]*m3[3] - m0[0]*m1[2]*m3[3]) /det,
            ( m0[2]*m1[3]*m2[0] - m0[3]*m1[2]*m2[0] + m0[3]*m1[0]*m2[2] - m0[0]*m1[3]*m2[2] - m0[2]*m1[0]*m2[3] + m0[0]*m1[2]*m2[3]) /det],
           [( m1[1]*m2[3]*m3[0] - m1[3]*m2[1]*m3[0] + m1[3]*m2[0]*m3[1] - m1[0]*m2[3]*m3[1] - m1[1]*m2[0]*m3[3] + m1[0]*m2[1]*m3[3]) /det,
            ( m0[3]*m2[1]*m3[0] - m0[1]*m2[3]*m3[0] - m0[3]*m2[0]*m3[1] + m0[0]*m2[3]*m3[1] + m0[1]*m2[0]*m3[3] - m0[0]*m2[1]*m3[3]) /det,
            ( m0[1]*m1[3]*m3[0] - m0[3]*m1[1]*m3[0] + m0[3]*m1[0]*m3[1] - m0[0]*m1[3]*m3[1] - m0[1]*m1[0]*m3[3] + m0[0]*m1[1]*m3[3]) /det,
            ( m0[3]*m1[1]*m2[0] - m0[1]*m1[3]*m2[0] - m0[3]*m1[0]*m2[1] + m0[0]*m1[3]*m2[1] + m0[1]*m1[0]*m2[3] - m0[0]*m1[1]*m2[3]) /det],
           [( m1[2]*m2[1]*m3[0] - m1[1]*m2[2]*m3[0] - m1[2]*m2[0]*m3[1] + m1[0]*m2[2]*m3[1] + m1[1]*m2[0]*m3[2] - m1[0]*m2[1]*m3[2]) /det,
            ( m0[1]*m2[2]*m3[0] - m0[2]*m2[1]*m3[0] + m0[2]*m2[0]*m3[1] - m0[0]*m2[2]*m3[1] - m0[1]*m2[0]*m3[2] + m0[0]*m2[1]*m3[2]) /det,
            ( m0[2]*m1[1]*m3[0] - m0[1]*m1[2]*m3[0] - m0[2]*m1[0]*m3[1] + m0[0]*m1[2]*m3[1] + m0[1]*m1[0]*m3[2] - m0[0]*m1[1]*m3[2]) /det,
            ( m0[1]*m1[2]*m2[0] - m0[2]*m1[1]*m2[0] + m0[2]*m1[0]*m2[1] - m0[0]*m1[2]*m2[1] - m0[1]*m1[0]*m2[2] + m0[0]*m1[1]*m2[2]) /det]]

def get_bounding_box(scene):
    bb_min = [1e10, 1e10, 1e10] # x,y,z
    bb_max = [-1e10, -1e10, -1e10] # x,y,z
    inv = numpy.linalg.inv if numpy else _inv
    return get_bounding_box_for_node(scene.rootnode, bb_min, bb_max, inv(scene.rootnode.transformation))

def get_bounding_box_for_node(node, bb_min, bb_max, transformation):

    if numpy:
        transformation = numpy.dot(transformation, node.transformation)
    else:
        t0,t1,t2,t3 = transformation
        T0,T1,T2,T3 = node.transformation
        transformation = [ [
                t0[0]*T0[0] + t0[1]*T1[0] + t0[2]*T2[0] + t0[3]*T3[0],
                t0[0]*T0[1] + t0[1]*T1[1] + t0[2]*T2[1] + t0[3]*T3[1],
                t0[0]*T0[2] + t0[1]*T1[2] + t0[2]*T2[2] + t0[3]*T3[2],
                t0[0]*T0[3] + t0[1]*T1[3] + t0[2]*T2[3] + t0[3]*T3[3]
            ],[
                t1[0]*T0[0] + t1[1]*T1[0] + t1[2]*T2[0] + t1[3]*T3[0],
                t1[0]*T0[1] + t1[1]*T1[1] + t1[2]*T2[1] + t1[3]*T3[1],
                t1[0]*T0[2] + t1[1]*T1[2] + t1[2]*T2[2] + t1[3]*T3[2],
                t1[0]*T0[3] + t1[1]*T1[3] + t1[2]*T2[3] + t1[3]*T3[3]
            ],[
                t2[0]*T0[0] + t2[1]*T1[0] + t2[2]*T2[0] + t2[3]*T3[0],
                t2[0]*T0[1] + t2[1]*T1[1] + t2[2]*T2[1] + t2[3]*T3[1],
                t2[0]*T0[2] + t2[1]*T1[2] + t2[2]*T2[2] + t2[3]*T3[2],
                t2[0]*T0[3] + t2[1]*T1[3] + t2[2]*T2[3] + t2[3]*T3[3]
            ],[
                t3[0]*T0[0] + t3[1]*T1[0] + t3[2]*T2[0] + t3[3]*T3[0],
                t3[0]*T0[1] + t3[1]*T1[1] + t3[2]*T2[1] + t3[3]*T3[1],
                t3[0]*T0[2] + t3[1]*T1[2] + t3[2]*T2[2] + t3[3]*T3[2],
                t3[0]*T0[3] + t3[1]*T1[3] + t3[2]*T2[3] + t3[3]*T3[3]
            ] ]

    for mesh in node.meshes:
        for v in mesh.vertices:
            v = transform(v, transformation)
            bb_min[0] = min(bb_min[0], v[0])
            bb_min[1] = min(bb_min[1], v[1])
            bb_min[2] = min(bb_min[2], v[2])
            bb_max[0] = max(bb_max[0], v[0])
            bb_max[1] = max(bb_max[1], v[1])
            bb_max[2] = max(bb_max[2], v[2])


    for child in node.children:
        bb_min, bb_max = get_bounding_box_for_node(child, bb_min, bb_max, transformation)

    return bb_min, bb_max

def search_library():
    '''
    Loads the assimp library.
    Throws exception AssimpError if no library_path is found

    Returns: tuple, (load from filename function,
                     load from memory function,
                     export to filename function,
                     export to blob function,
                     release function,
                     dll)
    '''

    # silence 'DLL not found' message boxes on win
    try:
        ctypes.windll.kernel32.SetErrorMode(0x8007)
    except AttributeError:
        pass

    libassimp = 'libassimp.so.5'
    LIBASSIMP = ctypes.CDLL(libassimp)
    try:
        load = LIBASSIMP.aiImportFile
        load_mem = LIBASSIMP.aiImportFileFromMemory
        export = LIBASSIMP.aiExportScene
        export2blob = LIBASSIMP.aiExportSceneToBlob
        release = LIBASSIMP.aiReleaseImport
    except AttributeError:
        # OK, this is a library, but it does not have the functions we need
        raise AssimpError("assimp library '%s' not found" % (libassimp,))
    else:
        # Library found!
        from . import structs
        load.restype = ctypes.POINTER(structs.Scene)
        load_mem.restype = ctypes.POINTER(structs.Scene)
        export2blob.restype = ctypes.POINTER(structs.ExportDataBlob)
    return(load, load_mem, export, export2blob, release, LIBASSIMP)

def hasattr_silent(object, name):
    """
        Calls hasttr() with the given parameters and preserves the legacy (pre-Python 3.2)
        functionality of silently catching exceptions.

        Returns the result of hasatter() or False if an exception was raised.
    """

    try:
        if not object:
            return False
        return hasattr(object, name)
    except AttributeError:
        return False
