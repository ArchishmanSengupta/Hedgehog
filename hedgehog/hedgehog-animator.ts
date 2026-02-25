/**
 * Hedgehog Animation Renderer
 * Following the technique from GitHub Copilot CLI's animated ASCII banner
 * https://github.blog/engineering/from-pixels-to-characters-the-engineering-behind-github-copilot-clis-animated-ascii-banner/
 */

import * as readline from 'readline';
import { hedgehogFrames, colorThemes } from './hedgehog-frames';

interface RenderOptions {
    theme?: 'dark' | 'light';
    loop?: boolean;
    fps?: number;
}

class HedgehogAnimator {
    private frames = hedgehogFrames;
    private currentFrame = 0;
    private intervalId: NodeJS.Timeout | null = null;
    private theme: 'dark' | 'light';
    private loop: boolean;
    private fps: number;

    constructor(options: RenderOptions = {}) {
        this.theme = options.theme || 'dark';
        this.loop = options.loop ?? true;
        this.fps = options.fps || 10;
    }

    /**
     * Apply color theming to the frame content
     */
    private applyTheme(content: string): string {
        const theme = colorThemes[this.theme];

        let colored = content;
        // Apply colors to specific characters
        colored = colored.replace(/[■□]/g, `${theme.quills}$&${theme.reset}`);
        colored = colored.replace(/[@]/g, `${theme.body}$&${theme.reset}`);
        colored = colored.replace(/[v~^u]/g, `${theme.nose}$&${theme.reset}`);
        colored = colored.replace(/[-]/g, `${theme.eyes}$&${theme.reset}`);
        colored = colored.replace(/[╭╮╰╯─]/g, `${theme.border}$&${theme.reset}`);

        return colored;
    }

    /**
     * Move cursor to top-left of terminal
     */
    private cursorToTop(): void {
        readline.cursorTo(process.stdout, 0, 0);
    }

    /**
     * Clear the screen below the cursor
     */
    private clearScreen(): void {
        readline.clearScreenDown(process.stdout);
    }

    /**
     * Render the current frame
     */
    private render(): void {
        const frame = this.frames[this.currentFrame];
        const themedContent = this.applyTheme(frame.content);

        // Move cursor to top-left and clear screen (prevents flicker)
        this.cursorToTop();
        this.clearScreen();

        // Write the frame
        process.stdout.write(themedContent);

        // Advance to next frame
        this.currentFrame = (this.currentFrame + 1) % this.frames.length;

        // If not looping and we've reached the end, stop
        if (!this.loop && this.currentFrame === 0) {
            this.stop();
        }
    }

    /**
     * Start the animation
     */
    start(): void {
        // Calculate interval from fps
        const intervalMs = 1000 / this.fps;

        // Initial render
        this.render();

        // Start animation loop
        this.intervalId = setInterval(() => {
            this.render();
        }, intervalMs);
    }

    /**
     * Stop the animation
     */
    stop(): void {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }

    /**
     * Run animation once (non-looping)
     */
    runOnce(): void {
        this.loop = false;
        this.start();
    }
}

/**
 * Main function to display the hedgehog animation
 */
export function showHedgehog(options: RenderOptions = {}): void {
    const animator = new HedgehogAnimator(options);
    animator.start();

    // Handle cleanup on exit
    process.on('SIGINT', () => {
        animator.stop();
        process.exit(0);
    });

    process.on('SIGTERM', () => {
        animator.stop();
        process.exit(0);
    });
}

export { HedgehogAnimator };
