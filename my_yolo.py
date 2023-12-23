from ultralytics import YOLO

class MyYOLO(YOLO):
    def track(self, source=None, stream=False, persist=True, **kwargs):
        """
        Perform object tracking on the input source using the registered trackers.

        Args:
            source (str, optional): The input source for object tracking. Can be a file path or a video stream.
            stream (bool, optional): Whether the input source is a video stream. Defaults to False.
            persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
            **kwargs (optional): Additional keyword arguments for the tracking process.

        Returns:
            (List[ultralytics.engine.results.Results]): The tracking results.
        """
        if not hasattr(self.predictor, 'trackers'):
            from trackers.track import register_tracker
            register_tracker(self, persist)
        kwargs['conf'] = kwargs.get('conf') or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs['mode'] = 'track'
        return super().predict(source=source, stream=stream, **kwargs)