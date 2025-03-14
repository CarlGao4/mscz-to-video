from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFontMetrics, QPainter
from PySide6.QtWidgets import QLabel, QSizePolicy, QStyle, QStyleOption


class TextWrappedQLabel(QLabel):
    """A QLabel that wraps text everywhere, not just at word boundaries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.textalignment = Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWrapAnywhere
        self.isTextLabel = True
        self.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        self._minimum_height = 0

    def paintEvent(self, event):
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)

        self.style().drawPrimitive(QStyle.PrimitiveElement.PE_Widget, opt, painter, self)

        self.style().drawItemText(painter, self.rect(), self.textalignment, self.palette(), True, self.text())

    def setMinimumHeight(self, height):
        self._minimum_height = height

    def heightForWidth(self, width):
        metrics = QFontMetrics(self.font())

        return metrics.boundingRect(0, 0, width, 0, self.textalignment, self.text()).height()

    def sizeHint(self):
        return QSize(self.width(), max(self._minimum_height, self.heightForWidth(self.width())))

    def minimumSizeHint(self):
        return QSize(0, 0)

    def resizeEvent(self, event):
        self.updateGeometry()
