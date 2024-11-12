///////////////////////
// DOM nodes
const $target_canvas = document.getElementById('target-canvas');
const $debug = document.getElementById('debug');

addEventListener("TrunkApplicationStarted", (event) => {

  //console.log("application started - bindings:", window.wasmBindings, "WASM:", event.detail.wasm);
  let { JsSystem } = window.wasmBindings;

  let app = null;

  window.reset_app = () => {
    app = {};
    render(app);
  }

  function render(app) {
    $debug.innerHTML = JSON.stringify(app)
  }

  reset_app();

});
