
## holds some constant value
## for rendering
MAZE_Large_str_maze_spec = \
    '############\\#OOOO#OOOOO#\\#O##O#O#O#O#\\#OOOOOO#OOO#\\#O####O###O#\\#OO#O#OOOOO#\\##O#O#O#O###\\#OO#OOO#OGO#\\############'

MAZE_Medium_str_maze_spec = \
    '########\\#OO##OO#\\#OO#OOO#\\##OOO###\\#OO#OOO#\\#O#OO#O#\\#OOO#OG#\\########'

MAZE_Umaze_str_maze_spec = '#####\\#GOO#\\###O#\\#OOO#\\#####'
## check back load env

def get_str_maze_spec(gr_e_name: str):
    """for gym robot env"""
    if gr_e_name == 'PointMaze_Large-v3':
        return MAZE_Large_str_maze_spec
    elif gr_e_name == 'PointMaze_Medium-v3':
        return MAZE_Medium_str_maze_spec
    elif gr_e_name == 'PointMaze_UMaze-v3':
        return MAZE_Umaze_str_maze_spec
    else:
        assert False
