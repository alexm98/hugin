import os
from tempfile import NamedTemporaryFile

import pytest

from hugin.engine.core import IdentityModel, AverageMerger, NullMerger
from hugin.engine.scene import RasterSceneTrainer, AvgEnsembleScenePredictor, RasterScenePredictor, \
    RasterIOSceneExporter
from hugin.io.loader import BinaryCategoricalConverter
from tests.conftest import generate_filesystem_loader


#@pytest.fixture
#def small_generated_filesystem_loader():
#    return generate_filesystem_loader(num_images=4, width=500, height=510)


# @pytest.mark.skipif(not runningInCI(), reason="Skipping running locally as it might be too slow")
def test_identity_train_complete_flow(generated_filesystem_loader):
    mapping = {
        'inputs': {
            'input_1': {
                'primary': True,
                'channels': [
                    ["RGB", 1],
                    ["RGB", 2],
                    ["RGB", 3]
                ]
            }
        },
        'target': {
            'output_1': {
                'channels': [
                    ["GTI", 1]
                ],
                'window_shape': (256, 256),
                'stride': 256,
                'preprocessing': [
                    BinaryCategoricalConverter(do_categorical=False)
                ]
            }
        }
    }

    with NamedTemporaryFile(delete=False) as named_temporary_file:
        named_tmp = named_temporary_file.name
        os.remove(named_temporary_file.name)
        identity_model = IdentityModel(name="dummy_identity_model", num_loops=3)
        trainer = RasterSceneTrainer(name="test_raster_trainer",
                                     stride_size=256,
                                     window_size=(256, 256),
                                     model=identity_model,
                                     mapping=mapping,
                                     destination=named_tmp)

        dataset_loader, validation_loader = generated_filesystem_loader.get_dataset_loaders()
        loop_dataset_loader_old = dataset_loader.loop
        loop_validation_loader_old = validation_loader.loop

        try:
            dataset_loader.loop = True
            validation_loader.loop = True
            print("Training on %d datasets" % len(dataset_loader))
            print("Using %d datasets for validation" % len(validation_loader))

            trainer.train_scenes(dataset_loader, validation_scenes=validation_loader)
            trainer.save()

            assert os.path.exists(named_tmp)
            assert os.path.getsize(named_tmp) > 0
        finally:
            dataset_loader.loop = loop_dataset_loader_old
            validation_loader.loop = loop_validation_loader_old


        new_mapping = mapping.copy()
        del new_mapping['target']

        raster_predictor = RasterScenePredictor(
            name="simple_raster_scene_predictor",
            model=identity_model,
            stride_size=256,
            window_size=(256, 256),
            mapping=new_mapping,
            #prediction_merger=AverageMerger,
            prediction_merger=NullMerger,
            post_processors=[]
        )
        avg_predictor = AvgEnsembleScenePredictor(
            name="simple_avg_ensable",
            predictors=[
                {
                    'predictor': raster_predictor,
                    'weight': 1
                }
            ])


        raster_saver = RasterIOSceneExporter(destination="/tmp/predictions/",
                                             srs_source_component="RGB",
                                             rasterio_creation_options={
                                                 'blockxsize': 256,
                                                 'blockysize': 256
                                             }
        )

        dataset_loader.reset()
        raster_saver.flow_prediction_from_source(dataset_loader, raster_predictor)