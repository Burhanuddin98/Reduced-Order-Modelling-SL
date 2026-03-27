"""
Room Acoustics Simulation Engine
================================
Structure-preserving SEM solver with stable model order reduction.

Supports:
  - Rectangular and arbitrary 2D geometries (via Gmsh quad meshing)
  - PR, FI, and LR boundary conditions
  - FOM (full order) and ROM (reduced order) solvers
  - GPU acceleration via CuPy when available

Based on:
  Bonthu et al. (2026) - Stable MOR for Time-Domain Room Acoustics
  Sampedro Llopis et al. (2022) - Reduced Basis Methods for Room Acoustics

Quick start (arbitrary geometry)::

    from room_acoustics.geometry import l_shaped_room, generate_quad_mesh
    from room_acoustics.unstructured_sem import (
        UnstructuredQuadMesh2D, assemble_unstructured_2d_operators)
    from room_acoustics.solvers import fom_pphi, build_psd_basis, rom_pphi

    geom = l_shaped_room(6.0, 4.0, 3.0, 2.0)
    raw = generate_quad_mesh(geom, h_target=0.3)
    mesh = UnstructuredQuadMesh2D(raw['nodes'], raw['quads'],
                                  raw['boundary'], P=4)
    ops = assemble_unstructured_2d_operators(mesh)
    result = fom_pphi(mesh, ops, 'PR', {}, 1.5, 1.5, 0.3, 1e-5, 0.05,
                      rec_idx=mesh.nearest_node(4.0, 0.5))
"""
