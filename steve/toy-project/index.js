import { exportSVG } from "pantograph2d"
import { drawCircle } from "pantograph2d/drawShape"
import { Loop, CubicBezier } from "pantograph2d/models"

import optimjs from "optimization-js"

import fs from "node:fs"
import { fileURLToPath } from "node:url"

import { distance, tgt } from "./vectorOperations.js"
import { splitmix32 } from "./utils.js"

import { integrateSurface, integrateLength } from "./geomIntegrations.js"
import { mcmc } from './mcmc.js'

const random = splitmix32(12)

class Shape {
  constructor(points, tangents = null, tangentsMagnitude = null) {
    this.points = points
    this.tangents = tangents || [
      tgt(this.points[3], this.points[0], this.points[1]),
      tgt(this.points[0], this.points[1], this.points[2]),
      tgt(this.points[1], this.points[2], this.points[3]),
      tgt(this.points[2], this.points[3], this.points[0]),
    ]
    this.tangentsMagnitude = tangentsMagnitude || [
      distance(this.points[0], this.points[1]) / 3,
      distance(this.points[1], this.points[2]) / 3,
      distance(this.points[2], this.points[3]) / 3,
      distance(this.points[3], this.points[0]) / 3,
    ]

    this._figure = null
  }

  startControlPoint(curveIndex) {
    const cos = Math.cos(this.tangents[curveIndex])
    const sin = Math.sin(this.tangents[curveIndex])

    return [
      this.points[curveIndex][0] + cos * this.tangentsMagnitude[curveIndex],
      this.points[curveIndex][1] + sin * this.tangentsMagnitude[curveIndex],
    ]
  }

  endControlPoint(curveIndex) {
    const index = (curveIndex + 1) % 4
    const cos = Math.cos(this.tangents[index])
    const sin = Math.sin(this.tangents[index])

    return [
      this.points[index][0] - cos * this.tangentsMagnitude[curveIndex],
      this.points[index][1] - sin * this.tangentsMagnitude[curveIndex],
    ]
  }

  get figure() {
    if (this._figure) return this._figure

    const figure = new Loop(
      [
        new CubicBezier(
          this.points[0],
          this.points[1],
          this.startControlPoint(0),
          this.endControlPoint(0),
        ),
        new CubicBezier(
          this.points[1],
          this.points[2],
          this.startControlPoint(1),
          this.endControlPoint(1),
        ),
        new CubicBezier(
          this.points[2],
          this.points[3],
          this.startControlPoint(2),
          this.endControlPoint(2),
        ),
        new CubicBezier(
          this.points[3],
          this.points[0],
          this.startControlPoint(3),
          this.endControlPoint(3),
        ),
      ],
      { ignoreChecks: true },
    )

    this._figure = figure
    return figure
  }

  surface(gridSize = 100) {
    return integrateSurface(this.figure)
  }

  length() {
    return integrateLength(this.figure)
  }
}

export const dpnt = (point, radius = 0.05) => {
  return drawCircle(radius).translateTo(point)
}

function saveShape(shape, name = "shape", dir = ".") {
  const svg = exportSVG(
    [
      { shape: drawCircle(1), color: "red" },
      shape.figure,
      ...shape.points.map((point) => dpnt(point)),
    ],
    { margin: 2 },
  )
  const dirURL = new URL(dir, import.meta.url)

  if (!fs.existsSync(fileURLToPath(dirURL))) {
    fs.mkdirSync(fileURLToPath(dirURL), { recursive: true })
  }

  const fileURL = new URL(`${dir}/${name}.svg`, import.meta.url)
  fs.writeFileSync(fileURLToPath(fileURL), svg)
}

function chooseInitialPoints() {
  const angles = [
    Math.random() * 2 * Math.PI,
    Math.random() * 2 * Math.PI,
    Math.random() * 2 * Math.PI,
    Math.random() * 2 * Math.PI,
  ].sort()

  return angles.map((a) => [Math.cos(a), Math.sin(a)])
}

function optimise() {
  const points = chooseInitialPoints()

  const initValues = [
    tgt(points[3], points[0], points[1]),
    tgt(points[0], points[1], points[2]),
    tgt(points[1], points[2], points[3]),
    tgt(points[2], points[3], points[0]),
    distance(points[0], points[1]) / 3,
    distance(points[1], points[2]) / 3,
    distance(points[2], points[3]) / 3,
    distance(points[3], points[0]) / 3,
  ]

  const vectFcn = (vec) => {
    const tgts = vec.slice(0, 4)
    for (let i = 0; i < 4; i++) {
      if (Math.abs(tgts[i]) > Math.PI)
        return -Infinity
    }
    const mags = vec.slice(4, 8)
    for (let i = 0; i < 4; i++) {
      if (mags[i] < 0)
        return -Infinity
    }

    const shape = new Shape(points, tgts, mags)
    const length = shape.length()
    const surface = shape.surface()

    const lengthErr = Math.pow(Math.log(length), 2) / 2
    const surfaceErr = Math.pow(surface - Math.PI, 2) / 0.2

    return lengthErr + surfaceErr
  }

  //  const result = mcmc(initValues, 100, 5, vectFcn)[0]
  const result = optimjs.minimize_Powell(vectFcn, initValues).argument

  return new Shape(
    points,
    result.slice(0, 4),
    result.slice(4, 8),
  )
}

function main() {
  const shape = optimise()
  console.log("area", shape.surface())
  console.log("length", shape.length())
  saveShape(shape)
}

main()
