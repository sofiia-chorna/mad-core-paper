Assignment of all 1.7 billion structures to their nearest MAD-CORE structure at every
level from 2^0 to 2^24. The nested ordering is used as a search tree (build_center_tree.py,
voronoi_net.py), assign_shards.py runs the descent over the shards and
record_populations counts the cell populations. cell_fps orders the members of every level 24 cell by FPS
seeded at the centre, stored cell after cell, with fps_order_in_cell equal to -1 for rows
that have no ordering of their own.
