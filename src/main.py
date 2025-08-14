import cv2

from sims.breakout import BreakoutSim


def main():
    sim = BreakoutSim()
    try:
        while True:
            # render current state
            sim.render()
            # read player input
            key = cv2.waitKey(16)
            if key == ord('q'):
                break
            if key == ord('a'):
                action = 1  # left
            elif key == ord('d'):
                action = 2  # right
            else:
                action = 0  # stay

            terminated = sim.tick(action)
            if terminated:
                sim = BreakoutSim()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
