from services.mini_serving.engine.serving_loop import ServingLoop
from services.mini_serving.engine.fake_engine import FakeEngine


def main():

    queue = ...
    scheduler = ...

    engine = FakeEngine()


    loop = ServingLoop(
        queue,
        scheduler,
        engine
    )


    loop.start()



if __name__=="__main__":
    main()
