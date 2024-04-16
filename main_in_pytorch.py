from dataclasses import dataclass
from functools import partial

import torch
import functorch

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('tkAgg')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


@dataclass
class System:
    grid: torch.Tensor
    cell_types: torch.Tensor
    types_as_grid: torch.Tensor

def setup(
    grid_size: tuple[int, int] = (100, 100),
    start_pos: tuple[int, int] = (20, 20),
    end_pos: tuple[int, int] = (80, 80),
    volume: float = 20,
) -> System:
    """Setup the simulation

    Set up a simulation with 50% of the grid between `start_pos` and `end_pos` filled
    with cells of type 1 and the other 50% with cells of type 2.
    """
    cell_size = int(torch.sqrt(torch.tensor(volume)))
    width = end_pos[0] - start_pos[0]
    height = end_pos[1] - start_pos[1]
    n_cells = int(width * height / cell_size**2)

    grid = torch.zeros(grid_size, dtype=torch.int8)
    types_as_grid = torch.zeros(grid_size, dtype=torch.int8)
    cell_types = torch.zeros(n_cells + 1, dtype=torch.int8)
    #cell_types[1:] = onp.random.choice([1, 2], size=n_cells)
    cell_types[1:] = torch.bernoulli(torch.empty(n_cells).uniform_(0, 1)) + 1
    cell_index = 1
    for x in range(start_pos[0], end_pos[0], cell_size):
        for y in range(start_pos[1], end_pos[1], cell_size):
            grid[x : x + cell_size, y : y + cell_size] = cell_index
            types_as_grid[x : x + cell_size, y : y + cell_size] = cell_types[cell_index]
            cell_index += 1

    return System(grid, cell_types,types_as_grid)

def cell_interaction_energy(
    cell_id: int,
    cell_type: int,
    grid: torch.tensor,
    cell_types: torch.tensor,
    types_as_grid: torch.tensor,
    J: torch.tensor,
    V: float,
    lambda_v: float,
    x: int,
    y: int,
) -> float:
    """Compute the interaction energy of a cell with its neighbours

    Args:
        cell_id: The id of the cell
        cell_type: The type of the cell
        grid: The current system state
        cell_types: The cell types of the cells in the system
        J: The interaction matrix between cell types
        V: The target volume of the cells
        lambda_v: The volume constraint strength
        x: The x coordinate of the cell
        y: The y coordinate of the cell
    """
    energy = 0
    #print(cell_types)
    #x_tensor = torch.tensor(x)
    #y_tensor = torch.tensor(y)
    #cell_id_tensor = cell_id.clone().detach()
    #grid_x_minus_1 = grid[x_tensor - 1, y]

    if x>0 and cell_id != grid[x-1,y]:
        energy+=J[cell_type, types_as_grid[x - 1, y]]

    if x < grid.shape[0] - 1 and cell_id!= grid[x + 1, y]:
        energy+=J[cell_type, types_as_grid[x + 1, y]]

    if x > 0 and cell_id != grid[x, y-1]:
        energy += J[cell_type, types_as_grid[x, y-1]]

    if x < grid.shape[0] - 1 and cell_id != grid[x, y+1]:
        energy += J[cell_type, types_as_grid[x, y+1]]

    # energy += torch.where(
    #
    #     torch.logical_and(x_tensor>0, cell_id_tensor != grid[x - 1, y]),
    #     J[cell_type, types_as_grid[x - 1, y]],
    #     0,
    # )
    # energy += torch.where(
    #     torch.logical_and(x_tensor < grid.shape[0] - 1, cell_id_tensor != grid[x + 1, y]),
    #     J[cell_type, types_as_grid[x + 1, y]],
    #     0,
    # )
    # energy += torch.where(
    #     torch.logical_and(y_tensor>0, cell_id_tensor != grid[x, y - 1]),
    #     J[cell_type, types_as_grid[x, y - 1]],
    #     0,
    # )
    # energy += torch.where(
    #     torch.logical_and(y_tensor < grid.shape[1] - 1, cell_id_tensor != grid[x, y + 1]),
    #     J[cell_type, types_as_grid[x, y + 1]],
    #     0,
    # )
    return energy


#@partial(jit, static_argnums=(5,))
def hamiltonian(
    grid: torch.tensor,
    cell_types: torch.tensor,
    types_as_grid: torch.tensor,
    J: torch.tensor,
    V: float,
    lambda_v: float,
    n_cells: int,
) -> float:
    """Compute the energy of the system

    Args:
        grid: The current system state
        cell_types: Mapping of cell-ids to cell types (i. e. 0, 1 or 2)
        J: The interaction matrix between cell types
        V: The target volume of the cells
        lambda_v: The volume constraint strength
        n_cells: The number of cells in the system
    """

    energy =0

    for x in range(0, len(grid[0])-1):
        for y in range(0, len(grid[1])-1):
            energy+= cell_interaction_energy(
                grid[x, y],
                types_as_grid[x, y],
                grid,
                cell_types,
                types_as_grid,
                J,
                V,
                lambda_v,
                x,y
            )


    return energy


def delta_energy(
    grid: torch.tensor,
    cell_types: torch.tensor,
    types_as_grid,
    flip_x: int,
    flip_y: int,
    new_cell_id: int,
    J: torch.tensor,
    V: float,
    lambda_v: float,
) -> float:
    """Compute the energy difference of the system after flipping a cell

    Args:
        grid: The current system state
        cell_types: The cell types of the cells in the system
        flip_x: The x coordinate of the cell to flip
        flip_y: The y coordinate of the cell to flip
        new_cell_id: The cell id of the cell to flip
        J: The interaction matrix between cell types
        V: The target volume of the cells
        lambda_v: The volume constraint strength
    """
    old_cell_id = grid[flip_x, flip_y]
    old_volume_old_cell = torch.sum(torch.where(grid == old_cell_id, 1, 0))
    new_volume_old_cell = old_volume_old_cell - 1
    old_volume_new_cell = torch.sum(torch.where(grid == new_cell_id, 1, 0))
    new_volume_new_cell = old_volume_new_cell + 1

    d_volume_energy_old_cell = torch.where(
        old_cell_id == 0,
        0,
        lambda_v * ((new_volume_old_cell - V) ** 2 - (old_volume_old_cell - V) ** 2),
    )
    d_volume_energy_new_cell = torch.where(
        new_cell_id == 0,
        0,
        lambda_v * ((new_volume_new_cell - V) ** 2 - (old_volume_new_cell - V) ** 2),
    )

    old_type = cell_types[grid[flip_x, flip_y]]
    new_type = cell_types[int(new_cell_id)]

    old_neighbour_energy = cell_interaction_energy(
        old_cell_id, old_type, grid, cell_types,types_as_grid, J, V, lambda_v, flip_x, flip_y
    )
    new_neighbour_energy = cell_interaction_energy(
        new_cell_id, new_type, grid, cell_types, types_as_grid,J, V, lambda_v, flip_x, flip_y
    )

    return (
        d_volume_energy_old_cell
        + d_volume_energy_new_cell
        + 2 * new_neighbour_energy
        - 2 * old_neighbour_energy
    )


def propose_flip(
    grid: torch.tensor, cell_types: torch.tensor
) -> tuple[int, int, int]:
    """Propose a cell to flip

    Args:
        key: The random number generator key
        grid: The current system state
        cell_types: The cell types of the cells in the system

    Returns:
        key: The random number generator key
        x: The x coordinate of the cell to flip
        y: The y coordinate of the cell to flip
        new_cell_id: The cell id of the cell to flip
    """
    # select a random cell
    x = -1
    y = -1
    new_cell_id = -1

    cell_id = -1
    neighbours = torch.tensor([-1, -1, -1, -1])

    while torch.all(neighbours == cell_id):
        x = torch.randint(size= (1,), low= 0, high = grid.shape[0]-1)[0]
        #key, subkey = torch.random.split(key)
        y = torch.randint(size= (1,), low= 0, high = grid.shape[1]-1)[0]

        cell_id = grid[x, y]
        # get surrounding cell types
        neighbours = torch.tensor(
            [
                grid[max(0, x - 1), y],
                grid[min(x + 1, grid.shape[0] - 1), y],
                grid[x, max(0, y - 1)],
                grid[x, min(y + 1, grid.shape[1] - 1)],
            ]
        )


    #u_neighbours = torch.unique(neighbours) #, size=4, fill_value=cell_id)
    #p = torch.where(u_neighbours != cell_id, 1, 0)
    p = torch.where(neighbours != cell_id, 1, 0)
    p = p/torch.sum(p)



    #u_neighbours = u_neighbours.clone().detach()
    #p = p.clone().detach()

    samples = torch.multinomial(p, 1, replacement=True)
    #new_cell_id = u_neighbours[samples]
    new_cell_id = neighbours[samples]

    return x, y, new_cell_id

def accept_flip(grid: torch.tensor, flip_x, flip_y, new_cell_id, types_as_grid, cell_types) -> torch.tensor:
    """Accept a proposed flip, i. e. alter the system state by flipping a cell

    Args:
        grid: The current system state
        flip_x: The x coordinate of the cell to flip
        flip_y: The y coordinate of the cell to flip
        new_cell_id: The cell id of the cell to flip
    """
    grid[flip_x, flip_y] = new_cell_id
    types_as_grid[flip_x,flip_y] = cell_types[int(new_cell_id)]

    return 0


def mc_sweep(
    grid: torch.tensor,
    energy: float,
    cell_types: torch.tensor,
    types_as_grid: torch.tensor,
    J: torch.tensor,
    V: float,
    lambda_v: float,
    temperature: float,
):
    """Perform a Monte-Carlo sweep

    Perform as many single Monte-Carlo steps as there are sites ("pixels") in the
    system.

    Args:
        key: The random number generator key
        grid: The current system state
        energy: The current value of the hamiltonian
        cell_types: Mapping of cell-ids to cell types (i. e. 0, 1 or 2)
        J: The interaction matrix between cell types
        V: The target volume of the cells
        lambda_v: The volume constraint strength
        temperature: The system temperature
    """

    def negative_de(x, y, new_cell_id, grid, de):
        return accept_flip(grid, x, y, new_cell_id, types_as_grid, cell_types), de

    def positive_de(x, y, new_cell_id, grid, de):

        randn = torch.rand(1) - torch.exp(-de / temperature)
        #if randn<0:
        #    return accept_flip(grid, x, y, new_cell_id, types_as_grid, cell_types), de
        return torch.where(
            randn < 0,
            accept_flip(grid, x, y, new_cell_id,types_as_grid, cell_types),
            grid,
        ), torch.where(randn < 0, de, 0)



    for i in range(0,grid.shape[0] * grid.shape[1]):
        # grid, energy = args
        x,y,new_cell_id = propose_flip(grid, cell_types)
        de = delta_energy(grid, cell_types, types_as_grid, x, y, new_cell_id, J, V, lambda_v)

        if de<0:
            negative_de(x,y,new_cell_id,grid,de)
        else:
            positive_de(x,y,new_cell_id,grid,de)
    return grid, types_as_grid


J = torch.tensor(
    [
        [0, 16, 16],
        [16, 2, 11],
        [16, 11, 16],
    ]
)  # Interaction matrix
V = 25  # Target volume
lambda_v = 2  # volume constraint
temperature = 10

system = setup()
n_cells = int(torch.max(system.grid))

energy = hamiltonian(system.grid, system.cell_types,system.types_as_grid, J, V, lambda_v, n_cells)
start_energy = energy

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
img = ax.imshow(system.types_as_grid)
# img = ax.imshow(np.asarray(system.grid))

for i in range(1000):
    system.grid, energy = mc_sweep(
        system.grid, energy, system.cell_types,system.types_as_grid, J, V, lambda_v, temperature
    )
    #img.set_data(system.types_as_grid)
    img.set_data(system.grid)
    # img.set_data(np.asarray(system.grid))
    ####fig.canvas.draw()
    fig.show()
    fig.canvas.flush_events()
    print(f"Energy: {energy}")

    print(f"Start energy: {start_energy}")
    print(f"Energy from de: {energy}")
    print(
    #f"Energy from hamiltonian: {hamiltonian(system.grid, system.cell_types,system.types_as_grid, J, V, lambda_v, n_cells)}"
    )
