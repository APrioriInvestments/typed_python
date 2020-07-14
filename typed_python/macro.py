import threading

from typed_python.type_function import ConcreteTypeFunction
from typed_python.compiler.python_ast_util import _linesCache

from keyword import iskeyword

_fileNameLock = threading.RLock()


class MacroFormatError(Exception):
    pass


class MacroNameError(NameError):
    pass


def isValidVariableName(string):
    return isinstance(string, str) and string.isidentifier() and not iskeyword(string)


def checkFormat(constructor):
    keyFormat = True
    if not isinstance(constructor, dict):
        keyFormat = False
    elif len(constructor) != 2:
        keyFormat = False
    elif "sourceText" not in constructor:
        keyFormat = False
    elif "locals" not in constructor:
        keyFormat = False
    if not keyFormat:
        raise MacroFormatError(
            "A macro function must return a dict with keys 'sourceText' and 'locals'"
            " and no other keys"
        )

    textFormat = True
    sourceText = constructor["sourceText"]
    if not isinstance(sourceText, list):
        textFormat = False
    else:
        if not len(sourceText):
            textFormat = False
        else:
            returnLine = sourceText[-1]
            if not returnLine[:7] == "return ":
                textFormat = False
        for line in sourceText:
            if not isinstance(line, str):
                textFormat = False
    if not textFormat:
        raise MacroFormatError(
            "The sourceText returned by a macro function must be a list of strings, the"
            " last being equal to f\"return {X}\" for some string X"
        )

    namespaceFormat = True
    namespace = constructor["locals"]
    if not isinstance(namespace, dict):
        namespaceFormat = False
    else:
        for key, value in namespace.items():
            if not isValidVariableName(key):
                namespaceFormat = False
    if not namespaceFormat:
        raise MacroFormatError(
            "The locals returned by a macro function must be a dict from valid variable "
            "names to accessible variables"
        )


def getSourceText(constructor):
    sourceText = constructor["sourceText"]
    return [sourceText[i] + "\n" for i in range(len(sourceText) - 1)]


def getReturnName(constructor):
    return constructor["sourceText"][-1][7:]


def getNamespace(constructor):
    return constructor["locals"]


class ConcreteMacro(ConcreteTypeFunction):
    def __init__(self, concreteMacro):
        self.count = 0
        self.concreteMacro = concreteMacro

        concreteTypeFunction = self.getConcreteTypeFunction()
        super().__init__(concreteTypeFunction)

    def getConcreteTypeFunction(self):
        def concreteTypeFunction(*args, **kwargs):
            constructor = self.concreteMacro(*args, **kwargs)
            checkFormat(constructor)
            sourceText = getSourceText(constructor)
            returnName = getReturnName(constructor)
            namespace = getNamespace(constructor)

            with _fileNameLock:
                filename = (
                    "macro stash at"
                    + self.concreteMacro.__code__.co_filename
                    + ", line "
                    + str(self.concreteMacro.__code__.co_firstlineno)
                    + ", unique call number "
                    + str(self.count)
                )
                self.count += 1

            code = compile("".join(sourceText), filename, "exec")

            if code.co_filename in _linesCache:
                raise Exception(
                    f"Filename collision on {code.co_filename}, review the macro code naming "
                    + "logic"
                )
            _linesCache[code.co_filename] = sourceText

            fake_globals = dict(self.concreteMacro.__globals__)
            fake_globals.update(namespace)

            try:
                exec(code, fake_globals)
            except NameError as e:
                msg = e.args[0]
                msg = msg.split(" ")
                varname = msg[1]
                varname = varname[1: len(varname) - 1]
                assert varname not in fake_globals
                raise MacroNameError(
                    f"Macro source text contains {varname} but it's not in the namespace"
                )
            try:
                return fake_globals[returnName]
            except KeyError as e:
                assert e.args[0] == returnName
                raise MacroNameError(
                    f"Macro source text tries to return {returnName} but does not define it"
                )

        return concreteTypeFunction


def Macro(f):
    """Decorate 'f' to be a 'Macro'.

    Like a 'TypeFunction', a 'Macro' takes a set of hashable arguments and produces a type
    object, and the result is memoized so that code in 'f' is executed only once for each
    distinct set of arguments.

    Unlike 'TypeFunction', the output of a call to 'f' is *not* the same as the output of a
    call to 'Macro(f)'. Instead, the output of a call to 'f' is a dictionary which provides
    string-based instructions for how to construct a type, and this type will be the output
    of a call to 'Macro(f)'.

    The format for the output of a call to 'f' is a dict with two keys, 'sourceText' and
    'locals', and corresponding values:

        'sourceText' --> a list of strings to be read as lines of source code
        'locals' --> a dict from variable names appearing in the source text to non-global
                variables in the namespace of f

    Responsibility for getting this right is on the user.

    Example:

    @Macro
    def f(T):
        names = T.ElementNames

        sourceText = []
        sourceText.append("@Entrypoint")
        sourceText.append("def f(x):")
        sourceText.append("    return T(")
        for name in names:
            sourceText.append(f"        {name}=x.{name},")
        sourceText.append("    )")
        sourceText.append("f_t = type(f)")
        sourceText.append("return f_t")

        return {
            'sourceText': sourceText,
            'locals': {"T": T},
        }

    The 'locals' field isn't restricted to input names to 'f', it can also contain variables
    defined in the body of f or in the scope where f is defined.
    """

    return ConcreteMacro(f)
