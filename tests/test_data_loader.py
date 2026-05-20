from data_loader import get_image_geo_record, get_image_path, list_sample_images


def test_list_sample_images_contains_known_files():
    images = list_sample_images()
    assert "train__t_10144.jpg" in images
    assert "val__v_611.jpg" in images


def test_get_image_path_resolves_existing_sample():
    image_path = get_image_path("train__t_10144.jpg")
    assert image_path.name == "train__t_10144.jpg"
    assert image_path.exists()


def test_get_image_geo_record_reads_expected_center():
    record = get_image_geo_record("train__t_10144.jpg")
    assert record["image_name"] == "t_10144.jpg"
    assert record["center"] == [118.1210570438674, 24.536394476542437]
