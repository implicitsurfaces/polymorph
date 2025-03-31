export function createMatches(constraints, seen) {
  const matches = [];

  return { matches, addToMatches };

  function addToMatches(ptId) {
    constraints.forEach((constraint) => {
      if (!constraint.points.includes(ptId)) return;
      if (
        !["fixed", "vertical", "horizontal", "coincident"].includes(
          constraint.name,
        )
      )
        return;

      const [p0, p1] = constraint.points;
      const otherPoint = p0 === ptId ? p1 : p0;

      if (constraint.name === "vertical") {
        let xMatch = `${otherPoint}.x`;
        addMatch(xMatch, null);
      }
      if (constraint.name === "horizontal") {
        let yMatch = `${otherPoint}.y`;
        addMatch(null, yMatch);
      }
      if (constraint.name === "coincident") {
        let xMatch = `${otherPoint}.x`;
        let yMatch = `${otherPoint}.y`;
        addMatch(xMatch, yMatch);
      }
      if (constraint.name === "fixed") {
        let xMatch = constraint.x;
        let yMatch = constraint.y;
        addMatch(xMatch, yMatch);
      }
    });

    function addMatch(xMatch, yMatch) {
      const x = `${ptId}.x`;
      const y = `${ptId}.y`;
      let xMatchGroup = matches.find((group) => group.has(x));
      let yMatchGroup = matches.find((group) => group.has(y));

      if (!xMatchGroup) {
        xMatchGroup = new Set([x]);
        matches.push(xMatchGroup);
      }
      if (!yMatchGroup) {
        yMatchGroup = new Set([y]);
        matches.push(yMatchGroup);
      }
      if (xMatch !== null) {
        xMatchGroup.add(xMatch);
      }
      if (yMatch !== null) {
        yMatchGroup.add(yMatch);
      }
    }

    // Merge overlapping sets in matches
    mergeOverlappingMatchSets();
    sortMatchSets(matches, seen);
  }

  // Helper function to merge any sets in the global `matches` array that share overlapping ids.
  function mergeOverlappingMatchSets() {
    let merged = true;
    while (merged) {
      merged = false;
      for (let i = 0; i < matches.length; i++) {
        for (let j = i + 1; j < matches.length; j++) {
          // If there is any overlap between matches[i] and matches[j]
          const setI = matches[i];
          const setJ = matches[j];
          const hasOverlap = [...setJ].some((id) => setI.has(id));
          if (hasOverlap) {
            // Merge setJ into setI
            for (const id of setJ) {
              setI.add(id);
            }
            // Remove setJ from matches
            matches.splice(j, 1);
            merged = true;
            j--; // Adjust index after removal
          }
        }
      }
    }
  }

  function sortMatchSets(matches, seen) {
    seen = new Set(seen);
    // Helper to check if a string represents a valid number
    function isNumeric(str) {
      return !isNaN(parseFloat(str)) && isFinite(str);
    }

    // Helper to check if item's base is in seen
    function baseInSeen(str) {
      if (typeof str !== "string") return false;
      const base = str.slice(0, -2); // Remove last 2 chars
      return seen.has(base);
    }

    // Process each match set
    for (let i = 0; i < matches.length; i++) {
      // Convert the set to an array
      const arr = Array.from(matches[i]);

      // Find numeric items in the array
      const numericItems = arr.filter((item) => isNumeric(item));
      if (numericItems.length > 1) {
        console.log("More than one number in set:", numericItems);
      }

      // Sort in priority order: numbers, seen bases, everything else
      arr.sort((a, b) => {
        const aIsNum = isNumeric(a);
        const bIsNum = isNumeric(b);
        const aInSeen = baseInSeen(a);
        const bInSeen = baseInSeen(b);

        if (aIsNum && !bIsNum) return -1;
        if (!aIsNum && bIsNum) return 1;
        if (aInSeen && !bInSeen) return -1;
        if (!aInSeen && bInSeen) return 1;
        return 0;
      });

      // Replace the set with the sorted array
      matches[i] = new Set(arr);
    }
  }
}
