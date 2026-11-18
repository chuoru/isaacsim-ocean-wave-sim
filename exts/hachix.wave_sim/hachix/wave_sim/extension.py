from __future__ import annotations

import omni.ext
import omni.kit.actions.core
import omni.usd
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items

from .examples.raft import RaftExample, build_raft_scene
from .ocean import Floater, Ocean

MENU = "Wave Sim"
ACTION = "create_ocean_and_raft"


class WaveSimExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._example: tuple[Ocean, Floater] | None = None
        omni.kit.actions.core.get_action_registry().register_action(
            ext_id, ACTION, lambda *_: self.create_raft_example(), display_name="Examples > Ocean + Raft"
        )
        self._menu = [
            MenuItemDescription(
                name="Examples", sub_menu=[MenuItemDescription(name="Ocean + Raft", onclick_action=(ext_id, ACTION))]
            )
        ]
        add_menu_items(self._menu, MENU)

    def create_raft_example(self, example: RaftExample | None = None) -> tuple[Ocean, Floater]:
        self._detach_example()
        self._example = build_raft_scene(omni.usd.get_context().get_stage(), example)
        return self._example

    def _detach_example(self) -> None:
        if self._example is not None:
            for part in self._example:
                part.detach()
        self._example = None

    def on_shutdown(self) -> None:
        self._detach_example()
        remove_menu_items(self._menu, MENU)
        omni.kit.actions.core.get_action_registry().deregister_action(self._ext_id, ACTION)
