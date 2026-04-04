"""
romacoustics — Reduced Order Modelling for Room Acoustics
==========================================================
Open-source implementation of the Laplace-domain Reduced Basis Method
from Sampedro Llopis et al. (2022), JASA 152(2), pp. 851-865.

Quick start:
    from romacoustics import Room

    room = Room.box_2d(2.0, 2.0)
    room.set_source(1.0, 1.0)
    room.set_receiver(0.2, 0.2)
    room.set_boundary_fi(Zs=5000)

    ir = room.solve(t_max=0.1)
    ir.plot()
    ir.to_wav('output.wav')
"""

from romacoustics.room import Room
from romacoustics.ir import ImpulseResponse

__version__ = '0.1.0'
__all__ = ['Room', 'ImpulseResponse']
