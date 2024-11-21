import JSZip from "jszip";
//import loadCode from "../../utils/loadCode";

const DEFAULT_SCRIPT = `
const { draw } = drawAPI;


export default () => {
  return draw([0, -0.5], 0.1)
    .line()
    .moveBy(-0.7, 0.8, 0.1)
    .arcFromStartControl([-0.2, 1.5])
    .horizontalMoveBy(0.7, 0.2)
    .arcFromStartControl([0.5, 1.5])
    .horizontalMoveBy(0.7, 0.1)
    .line()
    .close()
    .translate([0, -0.2])
    .shell(0.25);
};
`;

export const exportCode = async (rawCode) => {
  const zip = new JSZip();
  zip.file("code.js", rawCode);
  const content = await zip.generateAsync({
    type: "base64",
    compression: "DEFLATE",
    compressionOptions: {
      level: 6,
    },
  });
  const code = encodeURIComponent(content);

  const url = new URL(window.location);
  url.searchParams.set("code", code);

  return url.toString();
};

const getUrlParam = (paramName) => {
  const url = new URL(window.location);
  const urlParams = url.searchParams;

  const param = urlParams.get(paramName);

  if (!param) return;

  if (!urlParams.has("keep")) {
    urlParams.delete(paramName);
    window.history.pushState({}, "", url);
  }

  return param;
};

const getHashParam = (paramName) => {
  const url = new URL(window.location);
  const urlParams = new URLSearchParams(url.hash.substring(1));

  const param = urlParams.get(paramName);

  if (!param) return;

  if (!urlParams.has("keep")) {
    urlParams.delete(paramName);
    url.hash = urlParams.toString();
    window.history.pushState({}, "", url);
  }

  return param;
};

export default async function codeInit() {
  const fromUrl = getUrlParam("from-url");
  if (fromUrl) {
    try {
      //return await loadFromUrl(fromUrl);
    } catch (e) {
      console.error(e);
    }
  }

  const code = getHashParam("code") || getUrlParam("code");
  if (code) {
    try {
      //return await loadCode(code);
    } catch (e) {
      console.error(e);
    }
  }

  const defaultScript = localStorage.getItem("script") || DEFAULT_SCRIPT;
  return defaultScript;
}
