#   Copyright 2017 Braxton Mckee
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

class ConversionScopeInfo(object):
    def __init__(self, filename, line, col, types):
        object.__init__(self)

        self.filename = filename
        self.line = line
        self.col = col
        self.types = {k:v for k,v in types.iteritems() if isinstance(k,str) and v is not None}

    def __cmp__(self, other):
        return cmp(
            (self.filename,self.line,self.col,tuple(sorted(self.types.iteritems()))),
            (other.filename,other.line,other.col,tuple(sorted(other.types.iteritems())))
            )

    @staticmethod
    def CreateFromAst(ast, types):
        return ConversionScopeInfo(
            ast.filename,
            ast.line_number,
            ast.col_offset,
            dict(types)
            )

    def __str__(self):
        return "   File \"%s\", line %d\n%s" % (self.filename, self.line, 
            "".join(["\t\t%s=%s\n" % (k,v) for k,v in self.types.iteritems()])
            )

class ConversionException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)
        self.conversion_scope_infos=[]

    def add_scope(self, new_scope):
        if not self.conversion_scope_infos or self.conversion_scope_infos[0] != new_scope:
            self.conversion_scope_infos = [new_scope] + self.conversion_scope_infos

    def __str__(self):
        return self.message + "\n\n" + "".join(str(x) for x in self.conversion_scope_infos)

class UnassignableFieldException(ConversionException):
    def __init__(self, obj_type, attr, target_type):
        ConversionException.__init__(self, "missing attribute %s in type %s" % (attr,obj_type))
        
        self.obj_type = obj_type
        self.attr = attr
        self.target_type = target_type
