import { draw, drawCircle } from "../src/main";

let profile = drawCircle(0.3).translate([0.3, 0.5]);
profile = draw([0, -0.5])
  .line()
  .moveBy(-0.7, 0.6)
  .arcFromStartControl([-0.2, 1.3])
  .horizontalMoveBy(0.7)
  .arcFromStartControl([0.5, 1.3])
  .horizontalMoveBy(0.7)
  .line()
  .close()
  .morph(drawCircle(0.3).translate([0, -0.3]), 0.2);

profile.debugRender().then((image) => {
  console.log(image);
});
