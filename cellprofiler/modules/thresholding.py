# -*- coding: utf-8 -*-

"""

Thresholding

"""

import numpy
import skimage.exposure
import skimage.filters
import skimage.filters.rank
import skimage.morphology
import skimage.util

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting


class Thresholding(cellprofiler.module.ImageProcessing):
    module_name = "Thresholding"

    variable_revision_number = 1

    def create_settings(self):
        super(Thresholding, self).create_settings()

        self.local = cellprofiler.setting.Binary(
            u"Local",
            False
        )

        self.local_operation = cellprofiler.setting.Choice(
            u"Operation",
            [
                u"Adaptive",
                u"Otsu’s method",
                u"Percentile"
            ],
            u"Adaptive"
        )

        self.global_operation = cellprofiler.setting.Choice(
            u"Operation",
            [
                u"Iterative selection thresholding",
                u"Manual",
                u"Minimum cross entropy thresholding",
                u"Otsu’s method",
                u"Yen’s method"
            ],
            u"Minimum cross entropy thresholding"
        )

        self.block_size = cellprofiler.setting.OddInteger(
            u"Block size",
            value=99
        )

        self.adaptive_method = cellprofiler.setting.Choice(
            u"Adaptive method",
            [
                u"Gaussian",
                u"Mean",
                u"Median"
            ],
            u"Mean"
        )

        self.sigma = cellprofiler.setting.Float(
            u"Sigma",
            1.0,
            minval=0.0
        )

        self.offset = cellprofiler.setting.Float(
            u"Offset",
            value=0.0
        )

        self.radius = cellprofiler.setting.Integer(
            u"Radius",
            8
        )

        self.bins = cellprofiler.setting.Integer(
            u"Bins",
            value=256
        )

        self.minimum = cellprofiler.setting.Float(
            u"Minimum",
            value=0.0
        )

        self.maximum = cellprofiler.setting.Float(
            u"Maximum",
            value=1.0
        )

    def settings(self):
        __settings__ = super(Thresholding, self).settings()

        return __settings__ + [
            self.adaptive_method,
            self.bins,
            self.block_size,
            self.global_operation,
            self.local_operation,
            self.maximum,
            self.minimum,
            self.offset,
            self.radius,
            self.sigma,
            self.x_name,
            self.y_name
        ]

    def visible_settings(self):
        __settings__ = super(Thresholding, self).visible_settings()

        settings = __settings__ + [
            self.local
        ]

        if self.local.value:
            settings = settings + [
                self.local_operation
            ]

            if self.local_operation.value == u"Adaptive":
                settings = settings + [
                    self.adaptive_method
                ]

                if self.adaptive_method == u"Gaussian":
                    settings = settings + [
                        self.sigma
                    ]

                settings = settings + [
                    self.block_size,
                    self.offset
                ]
            elif self.local_operation.value == u"Otsu’s method":
                settings = settings + [
                    self.radius
                ]
            elif self.local_operation.value == u"Percentile":
                settings = settings + [
                    self.radius
                ]
        else:
            settings = settings + [
                self.global_operation
            ]

            if self.global_operation.value in [
                u"Iterative selection thresholding",
                u"Otsu’s method",
                u"Yen’s method"
            ]:
                settings = settings + [
                    self.bins
                ]

            if self.global_operation.value == u"Manual":
                settings = settings + [
                    self.minimum,
                    self.maximum
                ]

        return settings

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        x_data = skimage.img_as_ubyte(x_data)

        if self.local.value:
            if self.local_operation.value == u"Adaptive":
                y_data = adaptive(
                    x_data,
                    self.adaptive_method.value.lower(),
                    self.block_size.value,
                    self.offset.value,
                    self.sigma.value
                )
            elif self.local_operation.value == u"Otsu’s method":
                y_data = local_otsu(x_data, self.radius.value)
            elif self.local_operation.value == u"Percentile":
                y_data = percentile(x_data, self.radius.value)
        else:
            if self.global_operation.value == u"Iterative selection thresholding":
                y_data = iterative_selection(x_data, self.bins.value)
            elif self.global_operation.value == u"Manual":
                y_data = manual(x_data, self.minimum.value, self.maximum.value)
            elif self.global_operation.value == u"Minimum cross entropy thresholding":
                y_data = minimum_cross_entropy(x_data)
            elif self.global_operation.value == u"Otsu’s method":
                y_data = otsu(x_data, self.bins.value)
            elif self.global_operation.value == u"Yen’s method":
                y_data = yen(x_data, self.bins.value)

        y = cellprofiler.image.Image(
            image=y_data,
            parent_image=x,
            dimensions=x.dimensions
        )

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = x.dimensions


def adaptive(data, method, block_size, offset, sigma=None):
    if data.ndim == 3:
        y_data = numpy.zeros_like(data, dtype=numpy.bool)

        for index, plane in enumerate(data):
            y_data[index] = skimage.filters.threshold_adaptive(
                image=plane,
                block_size=block_size,
                method=method,
                offset=offset,
                param=sigma
            )

        return y_data

    return skimage.filters.threshold_adaptive(
        image=data,
        block_size=block_size,
        method=method,
        offset=offset,
        param=sigma
    )


def iterative_selection(data, nbins):
    y_data = skimage.filters.threshold_isodata(image=data, nbins=nbins)

    return data >= y_data


def local_otsu(data, radius):
    disk = skimage.morphology.disk(radius)

    if data.ndim == 3:
        y_data = numpy.zeros_like(data)

        for index, plane in enumerate(data):
            y_data[index] = skimage.filters.rank.otsu(plane, disk)
    else:
        y_data = skimage.filters.rank.otsu(data, disk)

    return data >= y_data


def manual(data, minimum, maximum):
    data = skimage.img_as_float(data)

    data = skimage.exposure.rescale_intensity(data)

    y_data = numpy.zeros_like(data, numpy.bool)

    y_data[data > minimum] = True

    y_data[data < maximum] = False

    return y_data


def minimum_cross_entropy(data):
    y_data = skimage.filters.threshold_li(image=data)

    return data >= y_data


def otsu(data, nbins):
    y_data = skimage.filters.threshold_otsu(image=data, nbins=nbins)

    return data >= y_data


def percentile(data, radius):
    disk = skimage.morphology.disk(radius)

    if data.ndim == 3:
        y_data = numpy.zeros_like(data)

        for index, data in enumerate(data):
            y_data[index] = skimage.filters.rank.percentile(data, disk)
    else:
        y_data = skimage.filters.rank.percentile(data, disk)

    return data >= y_data


def yen(data, nbins):
    y_data = skimage.filters.threshold_yen(image=data, nbins=nbins)

    return data >= y_data