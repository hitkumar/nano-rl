from dataclasses import dataclass
from functools import cached_property

from nano_rl.trainer.config import ModelConfig
from nano_rl.trainer.world import get_world

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


@dataclass
class ParallelDims:
    """
    Organizes GPUs into multi-dimensional grid for distributed training.
    dp_replicate * dp_shard * cp * tp * pp == world_size

    dp_replicate controls the dp degree, dp_shard is fsdp degree.
    All gpus in (cp * tp * pp) process the same batch of data along different axis
    """

    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    ep: int
    world_size: int

    _world_mesh: DeviceMesh = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate = self.dp_replicate
        dp_shard = self.dp_shard
        cp = self.cp
        tp = self.tp
        pp = self.pp
        ep = self.ep

        for d in (dp_replicate, cp, tp, pp, ep):
            assert d >= 1, f"Parallelism should be >= 1, it is {d}"

        if dp_shard == -1:
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * cp * tp * pp)

        assert dp_shard >= 1, "dp shard must be >= 1"

        assert dp_replicate * dp_shard * cp * tp * pp == self.world_size, (
            f"dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp({cp}) * tp({tp}) * pp({pp}) != world_size({self.world_size})"
        )

        if ep > 1:
            assert ep % cp == 0
            assert (dp_shard * cp) % ep == 0

    def _build_mesh_without_ep(self) -> DeviceMesh:
        dims, names = [], []
        # the ordering here is quite important. It decides how GPUs are arranged in the mesh with various 4D parallelism techniques.
        for d, name in zip(
            [self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp],
            ["pp", "dp_replicate", "dp_shard", "cp", "tp"],
        ):
            if d > 1 or name == "dp_shard":
                dims.append(d)
                names.append(name)

        mesh = init_device_mesh("cuda", dims, mesh_dim_names=names)
        # indicates dimensions used in data parallel
        dp_mesh_dim_names = []
        # dimensions across which params are shareded
        dp_shard_cp_mesh_dim_names = []
        # dimensions across which loss is computed
        dp_cp_mesh_dim_names = []

        if self.dp_replicate > 1:
            dp_mesh_dim_names.append("dp_replicate")
            dp_cp_mesh_dim_names.append("dp_replicate")

        # Different values of dp_shard actually see different mini batches of data
        dp_mesh_dim_names.append("dp_shard")
        dp_shard_cp_mesh_dim_names.append("dp_shard")
        dp_cp_mesh_dim_names.append("dp_shard")

        if self.cp > 1:
            dp_shard_cp_mesh_dim_names.append("cp")
            dp_cp_mesh_dim_names.append("cp")

        mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")
        mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_cp")
        return mesh

    def _build_mesh_with_ep(self) -> DeviceMesh:
        # number of dp_shard rows in each EP group
        dp_shard_in_ep = self.ep // self.cp
        # number of EP groups
        dp_shard_mod_ep = self.dp_shard // dp_shard_in_ep
        dims, names = [], []
        for d, name in zip(
            [
                self.pp,
                self.dp_replicate,
                dp_shard_mod_ep,
                dp_shard_in_ep,
                self.cp,
                self.tp,
            ],
            ["pp", "dp_replicate", "dp_shard_mod_ep", "dp_shard_in_ep", "cp", "tp"],
        ):
            # dp_shard_mod_ep is needed even if it's 1, whose FSDP wrapping
            # helps the MoE layers do mixed precision training
            if d > 1 or name == "dp_shard_mod_ep":
                dims.append(d)
                names.append(name)

        mesh = init_device_mesh("cuda", dims, mesh_dim_names=names)
        # create submeshes
        # indicates dimensions used in data parallel
        dp_mesh_dim_names = []
        # dimensions across which params are shareded
        dp_shard_cp_mesh_dim_names = []
        # dimensions across which loss is computed
        dp_cp_mesh_dim_names = []
        # mesh for expert all to all
        ep_mesh_all_dims = []

        if self.dp_replicate > 1:
            dp_mesh_dim_names.append("dp_replicate")
            dp_cp_mesh_dim_names.append("dp_replicate")

        # Different values of dp_shard_mod_ep actually see different mini batches of data
        dp_mesh_dim_names.append("dp_shard_mod_ep")
        dp_shard_cp_mesh_dim_names.append("dp_shard_mod_ep")
        dp_cp_mesh_dim_names.append("dp_shard_mod_ep")

        if "dp_shard_in_ep" in names:
            dp_mesh_dim_names.append("dp_shard_in_ep")
            dp_shard_cp_mesh_dim_names.append("dp_shard_in_ep")
            dp_cp_mesh_dim_names.append("dp_shard_in_ep")
            ep_mesh_all_dims.append("dp_shard_in_ep")

        if self.cp > 1:
            dp_shard_cp_mesh_dim_names.append("cp")
            dp_cp_mesh_dim_names.append("cp")
            ep_mesh_all_dims.append("cp")

        mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")
        mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_cp")
        if ep_mesh_all_dims:
            mesh[tuple(ep_mesh_all_dims)]._flatten(mesh_dim_name="ep")
        return mesh

    def build_mesh(self):
        if self.ep > 1:
            return self._build_mesh_with_ep()
        else:
            return self._build_mesh_without_ep()

    @property
    def world_mesh(self) -> DeviceMesh:
        # Lazy init here
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    @property
    def dp_enabled(self) -> bool:
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def cp_enabled(self):
        return self.cp > 1

    @property
    def dp_cp_enabled(self):
        return self.dp_enabled or self.cp_enabled

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def ep_enabled(self):
        return self.ep > 1

    @property
    def fsdp_enabled(self):
        return self.dp_shard_enabled or self.cp_enabled

    @cached_property
    def dp_degree(self):
        return self.dp_replicate * self.dp_shard

    @cached_property
    def non_data_parallel_size(self):
        return self.cp * self.tp * self.pp

    @cached_property
    def seq_len_divisor(self):
        # Sequence Parallel requires seq_len divisible by TP degree
        # Context Parallel requires seq_len divisible by 2 * CP degree
        # (when load balancing is enabled, which is the default)
        return self.tp * (self.cp * 2)


def get_parallel_dims(config: ModelConfig, seq_len: int | None = None) -> ParallelDims:
    parallel_dims = ParallelDims(
        dp_replicate=config.dp_replicate,
        dp_shard=-1,
        cp=config.cp,
        tp=config.tp,
        pp=1,
        ep=config.ep,
        world_size=get_world().world_size,
    )

    # Validate sequence length against parallel dimensions requirements
    if seq_len is not None and seq_len % parallel_dims.seq_len_divisor != 0:
        raise ValueError(
            f"Sequence length ({seq_len}) must be divisible by "
            f"seq_len_divisor ({parallel_dims.seq_len_divisor}) for the given parallel dimensions. "
            f"This requirement comes from context parallel (CP={config.cp}) and "
            f"tensor parallel (TP={config.tp}) configurations."
        )

    return parallel_dims
