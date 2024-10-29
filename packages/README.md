# TypeScript Packages

For now, these TypeScript packages are created via [Vite](https://vite.dev/).

## How to install Vite?

You don't really need to, it's taken care by npm. So you just need npm, for example on Ubuntu:

```
sudo apt install npm
```

## How to create a new package?

```
cd polymorph/packages
npm create vite@latest
```

Then when prompted interactively:
- enter the desired name of the new package
- choose a framework if any (e.g., React), otherwise Vanilla
- choose the TypeScript variant

## How to try the package locally?

```
cd polymorph/packages/foo
npm install
npm run dev
```

Then open the shown link in a browser.
