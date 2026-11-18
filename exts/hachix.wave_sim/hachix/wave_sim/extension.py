from __future__ import annotations

import omni.ext
import omni.kit.actions.core
import omni.usd
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items

from .scene import WaveSimConfig, WaveSimScene

MENU = "Wave Sim"
ACTION = "create_ocean_and_raft"


class WaveSimExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._scene: WaveSimScene | None = None
        omni.kit.actions.core.get_action_registry().register_action(
            ext_id, ACTION, lambda *_: self.create_scene(), display_name="Create Ocean + Raft"
        )
        self._menu = [MenuItemDescription(name="Create Ocean + Raft", onclick_action=(ext_id, ACTION))]
        add_menu_items(self._menu, MENU)

    def create_scene(self, config: WaveSimConfig | None = None) -> WaveSimScene:
        """Author the ocean and raft on the open stage. Press Play to animate."""
        if self._scene is not None:
            self._scene.detach()
        self._scene = WaveSimScene(config or WaveSimConfig())
        self._scene.build(omni.usd.get_context().get_stage())
        self._scene.attach()
        return self._scene

    def on_shutdown(self) -> None:
        if self._scene is not None:
            self._scene.detach()
        remove_menu_items(self._menu, MENU)
        omni.kit.actions.core.get_action_registry().deregister_action(self._ext_id, ACTION)
