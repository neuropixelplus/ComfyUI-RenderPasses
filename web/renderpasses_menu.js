// RenderPasses: adds a "RenderPasses" button to ComfyUI that opens the app in a new tab.
import { app } from "../../scripts/app.js";

const PAGE = new URL("./renderpasses.html", import.meta.url).href;
const open = () => window.open(PAGE, "_blank", "noopener");

app.registerExtension({
  name: "RenderPasses.Launcher",
  // new frontend: a command + a button in the top menu bar
  commands: [{ id: "RenderPasses.Open", label: "Open RenderPasses", icon: "pi pi-images", function: open }],
  menuCommands: [{ path: ["Extensions"], commands: ["RenderPasses.Open"] }],
  actionBarButtons: [{ icon: "pi pi-images", label: "RenderPasses", tooltip: "Open RenderPasses (2D plates → compositing passes)", onClick: open }],
});
