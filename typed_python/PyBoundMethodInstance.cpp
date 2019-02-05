#include "PyBoundMethodInstance.hpp"

BoundMethod* PyBoundMethodInstance::type() {
    return (BoundMethod*)extractTypeFrom(((PyObject*)this)->ob_type);
}
