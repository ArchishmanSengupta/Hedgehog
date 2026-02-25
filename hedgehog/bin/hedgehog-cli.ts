#!/usr/bin/env node

/**
 * Hedgehog CLI - Simple ASCII Banner
 */

import { hedgehogBanner, getColoredBanner } from '../hedgehog-banner';

// Parse command line arguments
const args = process.argv.slice(2);
let useColor = true;

for (const arg of args) {
    if (arg === '--help' || arg === '-h') {
        console.log(`
Hedgehog CLI - ASCII Banner

Usage: hedgehog [options]

Options:
  --no-color    Disable colored output
  -h, --help    Show this help message
`);
        process.exit(0);
    }
    if (arg === '--no-color') {
        useColor = false;
    }
}

// Display the hedgehog banner
console.log(useColor ? getColoredBanner() : hedgehogBanner);
