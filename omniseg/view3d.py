"""VTK MRI anatomy and segmentation surfaces embedded in a native Qt workspace."""
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox, QSlider, QComboBox, QGroupBox
import vtkmodules.qt
vtkmodules.qt.PyQtImpl = 'PySide6'
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkRenderingVolumeOpenGL2
import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingAnnotation import vtkAnnotatedCubeActor
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from .core.surfaces import build_surfaces
from .core.anatomy import create_anatomy, set_anatomy_opacity


class SurfaceWorker(QThread):
    ready = Signal(object)
    failed = Signal(str)
    def __init__(self,mask,affine,labels):
        super().__init__();self.mask,self.affine,self.labels=mask,affine,labels
    def run(self):
        try:self.ready.emit(build_surfaces(self.mask,self.affine,self.labels))
        except Exception as error:self.failed.emit(str(error))


class SurfaceViewer(QWidget):
    def __init__(self,entries=(),opacities=None):
        super().__init__()
        self.entries=list(entries);self.label_opacity=dict(opacities or {})
        self.actors={};self.anatomy_actor=None;self.anatomy_key=None
        self.worker=None;self.pending=None;self.generation=0;self.initialized=False
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0);layout.setSpacing(0)
        self.vtk=QVTKRenderWindowInteractor(self);self.vtk.setMinimumSize(200,200)
        self.renderer=vtkRenderer();self.renderer.SetBackground(0,0,0)
        window=self.vtk.GetRenderWindow();window.AddRenderer(self.renderer)
        window.SetMultiSamples(0);window.SetAlphaBitPlanes(1)
        self.renderer.SetUseDepthPeeling(True);self.renderer.SetUseDepthPeelingForVolumes(True)
        # Depth peeling must stay on at all times, including while the camera is dragged:
        # it is what interleaves the translucent image volume with the translucent mask
        # surfaces. With it off the volume pass is drawn over the surfaces and washes the
        # mask out (and nested regions such as tumours inside the liver turn to fog).
        # Peel count and occlusion ratio barely change the frame time once a volume is
        # involved, so 8 is kept as a safe depth for a few nested surfaces.
        self.renderer.SetMaximumNumberOfPeels(8);self.renderer.SetOcclusionRatio(.2)
        interactor=window.GetInteractor()
        interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        interactor.SetDesiredUpdateRate(30);interactor.SetStillUpdateRate(0.5)
        # The desired update rate makes the volume mapper coarsen its sampling during a drag
        # (the plain vtkActor meshes are not LOD-aware, so it does not affect them).
        layout.addWidget(self.vtk,1)
        self.marker=None
        from vtkmodules.vtkRenderingCore import vtkTextActor
        title=vtkTextActor();title.SetInput('3D')
        prop=title.GetTextProperty();prop.SetFontSize(14);prop.SetBold(True);prop.SetColor(.71,.55,.95)
        title.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport();title.SetPosition(.02,.94)
        self.renderer.AddViewProp(title)
        # Settings live in the existing inspector, leaving the image area free of headers.
        self.settings=QGroupBox('Hiển thị 3D');controls=QVBoxLayout(self.settings);controls.setSpacing(10)

        def muted(text):
            widget=QLabel(text);widget.setObjectName('muted');widget.setWordWrap(True);return widget

        def switch_row(text):
            row=QHBoxLayout();row.setContentsMargins(0,0,0,0)
            text_label=QLabel(text);text_label.setStyleSheet('font-weight: 600;')
            row.addWidget(text_label);row.addStretch()
            toggle=QCheckBox();row.addWidget(toggle)
            controls.addLayout(row)
            return toggle,text_label

        self.show_anatomy,_=switch_row('Hiện toàn bộ ảnh');self.show_anatomy.setChecked(True)
        self.show_anatomy.setToolTip('Dựng thể tích từ cường độ ảnh đang chọn. Không phải mask phân đoạn.')
        self.show_anatomy.toggled.connect(self.update_appearance)
        controls.addWidget(muted('Độ đục ảnh'))
        self.brain_opacity=QSlider(Qt.Orientation.Horizontal);self.brain_opacity.setRange(1,100);self.brain_opacity.setValue(12)
        self.brain_opacity.valueChanged.connect(self.update_appearance);controls.addWidget(self.brain_opacity)
        controls.addWidget(muted('CÁC VÙNG TRONG MASK'))
        self.toggles={};self.colors={}
        self.region_box=QVBoxLayout();self.region_box.setSpacing(10);controls.addLayout(self.region_box)
        controls.addWidget(muted('Độ đục vùng mask'))
        self.opacity=QSlider(Qt.Orientation.Horizontal);self.opacity.setRange(10,100);self.opacity.setValue(85)
        self.opacity.valueChanged.connect(self.update_appearance);controls.addWidget(self.opacity)
        self.direction=QComboBox();self.direction.addItems(['Góc nhìn tự do','Trước · A','Sau · P','Trái · L','Phải · R','Trên · S'])
        self.direction.currentIndexChanged.connect(self.set_direction);controls.addWidget(self.direction)
        reset=QPushButton('Đặt lại góc nhìn');reset.clicked.connect(self.reset_camera);controls.addWidget(reset)
        self.message=muted('Mở ảnh hoặc nạp mask để xem 3D.');controls.addWidget(self.message)
        self.set_labels(self.entries)
        self.setToolTip('Chuột trái: xoay · Chuột giữa: di chuyển · Cuộn: zoom · Esc: về lát cắt')

    def set_labels(self,entries,opacities=None):
        """Rebuild the per-region switches for a (possibly different) task's labels.

        opacities: optional {label: factor 0..1} multiplying the global mask opacity, so a
        large outer region (liver, edema) can stay translucent over the regions inside it."""
        self.entries=list(entries);self.label_opacity=dict(opacities or {})
        while self.region_box.count():
            widget=self.region_box.takeAt(0).widget()
            if widget is not None:widget.hide();widget.setParent(None);widget.deleteLater()
        self.toggles={};self.colors={label:color for label,_,color in self.entries}
        for label,name,color in self.entries:
            row=QWidget();layout=QHBoxLayout(row);layout.setContentsMargins(0,0,0,0)
            text_label=QLabel(name);text_label.setStyleSheet(f'color: rgb({color[0]},{color[1]},{color[2]}); font-weight: 600;')
            toggle=QCheckBox();toggle.setChecked(True);toggle.setEnabled(False);toggle.toggled.connect(self.update_appearance)
            layout.addWidget(text_label);layout.addStretch();layout.addWidget(toggle)
            self.region_box.addWidget(row);self.toggles[label]=toggle
        self.set_input(None)

    def activate(self):
        if not self.initialized:
            self.vtk.Initialize()
            cube=vtkAnnotatedCubeActor()
            for method,text in [('SetXPlusFaceText','R'),('SetXMinusFaceText','L'),('SetYPlusFaceText','A'),('SetYMinusFaceText','P'),('SetZPlusFaceText','S'),('SetZMinusFaceText','I')]:getattr(cube,method)(text)
            cube.GetCubeProperty().SetColor(.25,.4,.48);cube.GetTextEdgesProperty().SetColor(.9,.95,1)
            self.marker=vtkOrientationMarkerWidget();self.marker.SetOrientationMarker(cube)
            self.marker.SetInteractor(self.vtk.GetRenderWindow().GetInteractor());self.marker.SetViewport(0,0,.14,.18)
            self.marker.SetEnabled(True);self.marker.InteractiveOff();self.initialized=True
        self.start_pending();self.reset_camera();self.render()

    def render(self):
        if self.initialized:self.vtk.GetRenderWindow().Render()

    def set_anatomy(self,volume,channel=0,modality=None):
        key=(id(volume),channel,modality) if volume is not None else None
        if key==self.anatomy_key:return
        if self.anatomy_actor is not None:self.renderer.RemoveVolume(self.anatomy_actor)
        self.anatomy_actor=None;self.anatomy_key=key
        self.show_anatomy.setEnabled(volume is not None)
        if volume is not None:
            self.anatomy_actor=create_anatomy(volume,channel,modality)
            self.renderer.AddVolume(self.anatomy_actor)
        self.update_appearance();self.reset_camera()

    def set_input(self,mask,affine=None,source=''):
        self.generation+=1
        for actor in self.actors.values():self.renderer.RemoveActor(actor)
        self.actors={};self.pending=None
        for check in self.toggles.values():check.setEnabled(False)
        if mask is None:
            self.message.setText('Đang xem ảnh. Nạp mask để thêm các vùng tổn thương.')
        else:
            self.settings.setToolTip(source)
            self.pending=(self.generation,mask,np.array(affine,copy=True));self.message.setText('Sẵn sàng dựng vùng mask…')
        self.render()
        if self.initialized and self.isVisible():self.start_pending()

    def start_pending(self):
        if self.worker is not None or self.pending is None:return
        generation,mask,affine=self.pending;self.pending=None
        self.worker=SurfaceWorker(mask,affine,tuple(self.colors));self.message.setText('Đang dựng bề mặt mask…')
        self.worker.ready.connect(lambda meshes:self.accept_meshes(generation,meshes))
        self.worker.failed.connect(lambda text:self.accept_error(generation,text))
        self.worker.finished.connect(self.worker_finished);self.worker.start()

    def worker_finished(self):
        old=self.worker;self.worker=None;old.deleteLater()
        if self.isVisible():self.start_pending()

    def accept_error(self,generation,text):
        if generation==self.generation:self.message.setText('Không dựng được mask: '+text)

    def accept_meshes(self,generation,meshes):
        if generation!=self.generation:return
        for actor in self.actors.values():self.renderer.RemoveActor(actor)
        self.actors={}
        for label,mesh in meshes.items():
            mapper=vtkPolyDataMapper();mapper.SetInputData(mesh);mapper.ScalarVisibilityOff()
            actor=vtkActor();actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*(np.array(self.colors[label],dtype=float)/255))
            actor.GetProperty().SetInterpolationToPhong();actor.GetProperty().SetSpecular(.2);actor.GetProperty().SetSpecularPower(25)
            self.renderer.AddActor(actor);self.actors[label]=actor
        for label,check in self.toggles.items():check.setEnabled(label in self.actors)
        self.update_appearance();self.reset_camera()
        self.message.setText('Kéo trái: xoay · Cuộn: zoom. Nguồn mask ở bảng thông tin bên dưới.' if meshes else 'Mask không có vùng nào của mô hình này; vẫn có thể xem ảnh.')

    def update_appearance(self,*args):
        for label,actor in self.actors.items():
            actor.SetVisibility(self.toggles[label].isChecked());actor.GetProperty().SetOpacity(self.opacity.value()/100*self.label_opacity.get(label,1.))
        if self.anatomy_actor:
            self.anatomy_actor.SetVisibility(self.show_anatomy.isChecked())
            set_anatomy_opacity(self.anatomy_actor,self.brain_opacity.value()/100)
        self.render()

    def reset_camera(self):
        if not self.actors and self.anatomy_actor is None:return
        bounds=self.renderer.ComputeVisiblePropBounds()
        if bounds[0]>bounds[1]:return
        center=np.array([(bounds[i]+bounds[i+1])/2 for i in (0,2,4)])
        extent=max(bounds[i+1]-bounds[i] for i in (0,2,4))
        camera=self.renderer.GetActiveCamera();camera.SetFocalPoint(*center)
        camera.SetPosition(*(center+np.array([1.,2.,1.])*max(extent,1)));camera.SetViewUp(0,0,1)
        self.renderer.ResetCamera();self.renderer.ResetCameraClippingRange()
        self.direction.blockSignals(True);self.direction.setCurrentIndex(0);self.direction.blockSignals(False);self.render()

    def set_direction(self,index):
        if not self.actors and self.anatomy_actor is None:return
        if index==0:self.reset_camera();return
        camera=self.renderer.GetActiveCamera();center=np.array(camera.GetFocalPoint());distance=camera.GetDistance()
        vector={1:(0,1,0),2:(0,-1,0),3:(-1,0,0),4:(1,0,0),5:(0,0,1)}[index]
        camera.SetPosition(*(center+np.array(vector)*distance));camera.SetViewUp(*((0,1,0) if index==5 else (0,0,1)))
        self.renderer.ResetCameraClippingRange();self.render()

    def is_busy(self):return self.worker is not None and self.worker.isRunning()
    def shutdown(self):
        if self.marker:self.marker.SetEnabled(False)
        self.vtk.Finalize()
