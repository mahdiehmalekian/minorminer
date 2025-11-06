from burnaby.cube_embedding._floor.cure import cure
from burnaby.cube_embedding._floor.trim import trim
from burnaby.cube_embedding._floor.uwj_floor import UWJFloor
from burnaby.cube_embedding._tile import PathInfo, VHIdxEdge
from burnaby.lattice_embedding import ZLatticeSurvey

__all__ = ["embed_on_floor_zpath"]


def _make_cube_chains(
    z_path: list[VHIdxEdge],
    uwj_floor: UWJFloor,
    Lx: int,
    Ly: int,
) -> dict[
    tuple[int, int, int], tuple[tuple[int, int, int, int, int], tuple[int, int, int, int, int]]
]:
    """Finds the cube chains constructed on a floor using the provided z-path.

    Args:
        z_path (list[VHIdxEdge]): The z-path within each floor's tile.
        uwj_floor (UWJFloor): The floor to construct the chains on.
        Lx (int): Dimension of cube on x-direction
        Ly (int): Dimension of cube on y-direction

    Returns:
        dict[tuple[int, int, int], tuple[tuple[int, int, int, int, int], tuple[int, int, int, int, int]]]:
        A dictionary where
            - keys are (x, y, z) corresponding to the nodes of the cube.
            - Values are pairs of 5-tuples cooresponding to the length-2 chains for each node, expressed
                in Zephyr coordinates.
    """
    col_row_nodes = uwj_floor.get_col_row_nodes(Lx=Lx, Ly=Ly)
    cube_chains = {}
    for (x, y), xy_nodes in col_row_nodes.items():
        for z, (a, b) in enumerate(z_path):
            cube_chains[(x, y, z)] = (xy_nodes[a].pop(0)).to_tuple(), (
                xy_nodes[b].pop(0)
            ).to_tuple()

    return cube_chains


def embed_on_floor_zpath(
    uwj_floor: UWJFloor,
    path_info: PathInfo,
    lattice_survey: ZLatticeSurvey,
    Lx: int,
    Ly: int,
) -> dict[
    tuple[int, int, int], tuple[tuple[int, int, int, int, int], tuple[int, int, int, int, int]]
]:
    """Attempts to find an embedding of the cube with given x- and y- dimension
        on the floor using the information of a z-path.

    Args:
        uwj_floor (UWJFloor): The floor to find the embedding on.
        path_info (PathInfo): The information of the path to use for constructing the z-paths.
        lattice_survey (ZLatticeSurvey): ZLatticeSurvey of the graph to find an embedding on.
        Lx (int): Dimension of cube on x-direction
        Ly (int): Dimension of cube on y-direction

    Returns:
        dict[tuple[int, int, int], tuple[tuple[int, int, int, int, int], tuple[int, int, int, int, int]]]:
            - A dictionary where
                - keys are (x, y, z) corresponding to the nodes of the cube.
                - Values are pairs of 5-tuples cooresponding to the length-2 chains for each node, expressed
                    in Zephyr coordinates.
    """
    working_uwj_floor = uwj_floor.copy()
    try:
        trim(
            uwj_floor=working_uwj_floor,
            path_info=path_info,
            lattice_survey=lattice_survey,
        )
        cure(
            uwj_floor=working_uwj_floor,
            zpath_info=path_info,
            lattice_survey=lattice_survey,
        )
        return _make_cube_chains(
            z_path=path_info.path_vh, uwj_floor=working_uwj_floor, Lx=Lx, Ly=Ly
        )
    except ValueError:
        return {}
