from restoration2 import RestorationEngine

engine = RestorationEngine(
    efd_harmonics=20,
    max_bridge_dist=0.25,   # 25 % of image diagonal
)
result = engine.restore("test_images/restoration_test.png", output_dir="restored/")
result.report()        # prints structured XAI report to stdout
result.save_visuals()  # writes PNG + JSON to output_dir