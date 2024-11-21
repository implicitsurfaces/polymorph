import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import React from "react";
import { setup } from "goober";

import "./index.css";

import App from "./App.tsx";

setup(React.createElement);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
