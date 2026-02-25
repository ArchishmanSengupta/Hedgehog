/**
 * Hedgehog ASCII Banner
 * Simple CLI banner marker
 */

export const hedgehogBanner = `
      /\\ /\\ /\\ /\\ /\\ /\\ /\\ /\\
    /\\ /\\ /\\ /\\ /\\ /\\ /\\ /\\ /\\ /\\
  /\\ /\\ /\\ /\\ /\\ /\\ /\\ /\\ /\\ /\\ /\\
 /\\ /\\ /\\ /\\ /\\ /\\ /\\ /\\ /\\ /\\ /\\ /\\
 ( ◕  ◉              )─╮
  \\    ▾                ___|
   \\________________________/
   ╰╯   ╰╯   ╰╯   ╰╯
`;

// Color-coded version
export const coloredHedgehogBanner = {
    quills: '\x1b[36m',   // Cyan
    face: '\x1b[35m',     // Magenta
    eyes: '\x1b[33m',    // Yellow
    body: '\x1b[37m',    // White
    reset: '\x1b[0m'
};

export function getColoredBanner(): string {
    const c = coloredHedgehogBanner;
    return hedgehogBanner
        .replace(/\//g, `${c.quills}$&${c.reset}`)
        .replace(/\\/g, `${c.quills}$&${c.reset}`)
        .replace(/◕|◉/g, `${c.face}$&${c.reset}`)
        .replace(/▾/g, `${c.eyes}$&${c.reset}`)
        .replace(/_/g, `${c.body}$&${c.reset}`)
        .replace(/╮|╰|╯/g, `${c.body}$&${c.reset}`);
}
