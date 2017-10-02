#!/usr/bin/python3

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

"""
This is the primary unit-test entrypoint for nativepython.
"""
import sys
import logging
import unittest
import os
import re
import time
import nose
import nose.config
import nose.loader
import nose.plugins.manager
import nose.plugins.xunit
import argparse
import nativepython
import nativepython.runtime as runtime
import nativepython_tests.test_config as test_config
import traceback

class DirectoryScope(object):
    def __init__(self, directory):
        self.directory = directory

    def __enter__(self):
        self.originalWorkingDir = os.getcwd()
        os.chdir(self.directory)

    def __exit__(self, exc_type, exc_value, traceback):
        os.chdir(self.originalWorkingDir)



def sortedBy(elts, sortFun):
    return [x[1] for x in sorted([(sortFun(y),y) for y in elts])]


def loadTestModules(testFiles, rootDir, rootModule):
    modules = set()
    for f in testFiles:
        try:
            with DirectoryScope(os.path.split(f)[0]):
                moduleName  = fileNameToModuleName(f, rootDir, rootModule)
                logging.info('importing module %s', moduleName)
                __import__(moduleName)
                modules.add(sys.modules[moduleName])
        except ImportError:
            logging.error("Failed to load test module: %s", moduleName)
            traceback.print_exc()
            raise

    return modules


def fileNameToModuleName(fileName, rootDir, rootModule):
    tr = (
        fileName
            .replace('.py', '')
            .replace(rootDir, '')
            .replace(os.sep, '.')
            )
    if tr.startswith('.'):
        return tr[1:]
    return tr

class OrderedFilterAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not 'ordered_actions' in namespace:
            setattr(namespace, 'ordered_actions', [])
        previous = namespace.ordered_actions
        previous.append((self.dest, values))
        setattr(namespace, 'ordered_actions', previous)

class PythonTestArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super(PythonTestArgumentParser,self).__init__(add_help = False)
        self.add_argument(
            '-v',
            dest='testHarnessVerbose',
            action='store_true',
            default=False,
            required=False,
            help="run test harness verbosely"
            )
        self.add_argument(
            '--list',
            dest='list',
            action='store_true',
            default=False,
            required=False,
            help="don't run tests, just list them"
            )
        self.add_argument(
            '--deep',
            dest='deep',
            action='store_true',
            default=False,
            required=False,
            help="run deeper level of testing in fuzztests"
            )
        self.add_argument(
            '--dump_llvm',
            dest='dump_llvm',
            action='store_true',
            default=False,
            required=False,
            help="dump llvm IR as it's produced"
            )
        self.add_argument(
            '--dump_type_signatures',
            dest='dump_type_signatures',
            action='store_true',
            default=False,
            required=False,
            help="dump type signatures of functions as they're produced"
            )
        self.add_argument(
            '--disable_optimization',
            dest='disable_optimization',
            action='store_true',
            default=False,
            required=False,
            help="disable optimization of llvm IR"
            )
        self.add_argument(
            '--dump_native',
            dest='dump_native',
            action='store_true',
            default=False,
            required=False,
            help="dump native ast code as it's produced"
            )
        self.add_argument('--filter',
                            nargs = 1,
                            help = 'restrict tests to a subset matching FILTER',
                            action = OrderedFilterAction,
                            default = None)
        self.add_argument('--add',
                            nargs = 1,
                            help = 'add back tests matching ADD',
                            action = OrderedFilterAction,
                            default = None)
        self.add_argument('--exclude',
                            nargs = 1,
                            help = "exclude python unittests matching 'regex'. "
                                  +"These go in a second pass after -filter",
                            action = OrderedFilterAction,
                            default = None)

    def parse_args(self,toParse):
        argholder = super(PythonTestArgumentParser,self).parse_args(toParse)

        args = None
        if 'ordered_actions' in argholder:
            args = []
            for arg,l in argholder.ordered_actions:
                args.append((arg,l[0]))
        return argholder, args

def regexMatchesSubstring(pattern, toMatch):
    for _ in re.finditer(pattern, toMatch):
        return True
    return False

def applyFilterActions(filterActions, tests):
    filtered = [] if filterActions[0][0] == 'add' else list(tests)

    for action, pattern in filterActions:
        if action == "add":
            filtered += [x for x in tests if
                                    regexMatchesSubstring(pattern, x.id())]
        elif action == "filter":
            filtered = [x for x in filtered if
                                    regexMatchesSubstring(pattern, x.id())]
        elif action == "exclude":
            filtered = [x for x in filtered if
                                    not regexMatchesSubstring(pattern, x.id())]
        else:
            assert False

    return filtered

def printTests(testCases):
    for test in testCases:
        print(test.id())

def runPyTestSuite(config, testFiles, testCasesToRun, testArgs):
    testProgram = nose.core.TestProgram(
        config=config,
        defaultTest=testFiles,
        suite=testCasesToRun,
        argv=testArgs,
        exit=False
        )

    return not testProgram.success

def loadTestsFromModules(config, modules):
    loader = nose.loader.TestLoader(config = config)
    allSuites = []
    for module in modules:
        cases = loader.loadTestsFromModule(module)
        allSuites.append(cases)

    return allSuites

def extractTestCases(suites):
    testCases = flattenToTestCases(suites)
    #make sure the tests are sorted in a sensible way.
    sortedTestCases = sortedBy(testCases, lambda x: x.id())

    return [x for x in sortedTestCases if not testCaseHasAttribute(x, 'disabled')]

def flattenToTestCases(suite):
    if isinstance(suite, list) or isinstance(suite, unittest.TestSuite):
        return sum([flattenToTestCases(x) for x in suite], [])
    return [suite]

def testCaseHasAttribute(testCase, attributeName):
    """Determine whether a unittest.TestCase has a given attribute."""
    if hasattr(getattr(testCase, testCase._testMethodName), attributeName):
        return True
    if hasattr(testCase.__class__, attributeName):
        return True
    return False


def loadTestCases(config, testFiles, rootDir, rootModule):
    modules = sortedBy(loadTestModules(testFiles, rootDir, rootModule), lambda module: module.__name__)
    allSuites = loadTestsFromModules(config, modules)
    return extractTestCases(allSuites)

def findTestFiles(rootDir, testRegex):
    logging.info('finding files from root %s', rootDir)
    testPattern = re.compile(testRegex)
    testFiles = []
    for directory, subdirectories, files in os.walk(rootDir):
        testFiles += [os.path.join(directory, f) for f in files if testPattern.match(f) is not None]

    return testFiles

def runPythonUnitTests(args, filter_actions):
    """run python unittests in all files in the 'tests' directory in the project.

    Args contains arguments from a UnitTestArgumentParser.

    Returns True if any failed.
    """
    root_dir = os.path.join(
        os.path.split(os.path.split(nativepython.__file__)[0])[0],
        "nativepython_tests"
        )

    return runPythonUnitTests_(
        args, filter_actions, testGroupName = "python",
        testFiles = findTestFiles(root_dir, '.*_test.py$')
        )

def logAsInfo(*args):
    if len(args) == 1:
        print(time.asctime(), " | ", args)
    else:
        print(time.asctime(), " | ", args[0] % args[1:])

def setLoggingLevel(level):
    logging.getLogger().setLevel(level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)


class OutputCapturePlugin(nose.plugins.base.Plugin):
    """
    Output capture plugin. Enabled by default. Disable with ``-s`` or
    ``--nocapture``. This plugin captures stdout during test execution,
    appending any output captured to the error or failure output,
    should the test fail or raise an error.
    """
    enabled = True
    name = 'OutputCaptureNosePlugin'
    score = 16010

    def __init__(self):
        self.stdout = []
        self.stdoutFD = None
        self.stderrFD = None
        self.fname = None
        self.hadError=False
        self.outfile = None
        self.testStartTime = None
        self.nocaptureall = False

    def options(self, parser, env):
        """Register commandline options
        """
        parser.add_option(
            "--nocaptureall", action="store_true",
            default=False, dest="nocaptureall"
            )

    def configure(self, options, conf):
        """Configure plugin. Plugin is enabled by default.
        """
        self.conf = conf
        if options.nocaptureall:
            self.nocaptureall = True

    def afterTest(self, test):
        """Clear capture buffer.
        """
        if self.nocaptureall:
            if not self.hadError:
                logAsInfo("\tpassed in %s", time.time() - self.testStartTime)
            else:
                logAsInfo("\tfailed in %s seconds. See logs in %s", time.time() - self.testStartTime, self.fname)


        if self.stdoutFD is None:
            return

        setLoggingLevel(logging.ERROR)

        sys.stdout.flush()
        sys.stderr.flush()

        os.dup2(self.stdoutFD, 1)
        os.close(self.stdoutFD)

        os.dup2(self.stderrFD, 2)
        os.close(self.stderrFD)

        self.stdoutFD = None
        self.stderrFD = None

        self.outfile.flush()
        self.outfile.close()
        self.outfile = None

        if not self.hadError:
            try:
                os.remove(self.fname)
            except OSError:
                pass
            logAsInfo("\tpassed in %s", time.time() - self.testStartTime)
        else:
            #the test failed. Report the failure
            logAsInfo("\tfailed in %s seconds. See logs in %s", time.time() - self.testStartTime, self.fname)


    def begin(self):
        pass

    def beforeTest(self, test):
        """Flush capture buffer.
        """
        logAsInfo("Running test %s", test)

        self.testStartTime = time.time()

        if self.nocaptureall:
            self.hadError=False
            return

        sys.stdout.flush()
        sys.stderr.flush()

        self.stdoutFD = os.dup(1)
        self.stderrFD = os.dup(2)

        self.fname = "nose.%s.%s.log" % (test.id(), os.getpid())

        if os.getenv("TEST_ERROR_OUTPUT_DIRECTORY", None) is not None:
            self.fname = os.path.join(os.getenv("TEST_ERROR_OUTPUT_DIRECTORY"), self.fname)

        self.outfile = open(self.fname, "w")

        os.dup2(self.outfile.fileno(), 1)
        os.dup2(self.outfile.fileno(), 2)

        self.hadError=False

        setLoggingLevel(logging.INFO)

    def formatError(self, test, err):
        """Add captured output to error report.
        """
        self.hadError=True

        if self.nocaptureall:
            return err

        ec, ev, tb = err

        ev = self.addCaptureToErr(ev, tb)

        #print statements here show up in the logfile
        print("Test ", test, ' failed: ')
        print(ev)

        self.failureReason = ev

        return (ec, ev, tb)

    def formatFailure(self, test, err):
        """Add captured output to failure report.
        """
        self.hadError=True
        return self.formatError(test, err)

    def addCaptureToErr(self, ev, tb):
        return ''.join([str(ev) + "\n"] + traceback.format_tb(tb) + ['\n>> output captured in %s <<' % self.fname])

    def end(self):
        pass

    def finalize(self, result):
        """Restore stdout.
        """


def runPythonUnitTests_(args, filterActions, testGroupName, testFiles):
    testArgs = ["dummy"]

    if args.deep:
        test_config.tests_are_deep = args.deep

    if args.dump_llvm:
        runtime.Runtime.singleton().compiler.mark_llvm_codegen_verbose()

    if args.dump_native:
        runtime.Runtime.singleton().compiler.mark_converter_verbose()

    if args.dump_type_signatures:
        runtime.Runtime.singleton().converter.verbose = True

    if args.disable_optimization:
        runtime.Runtime.singleton().compiler.optimize = False

    if args.testHarnessVerbose or args.list:
        testArgs.append('--nocapture')

    testArgs.append('--verbosity=0')

    if not args.list:
        print("Executing %s unit tests." % testGroupName)

    root_dir = os.path.split(os.path.split(nativepython.__file__)[0])[0]

    testCasesToRun = []

    plugins = nose.plugins.manager.PluginManager([OutputCapturePlugin()])

    config = nose.config.Config(plugins=plugins)
    config.configure(testArgs)
    
    testCases = loadTestCases(config, testFiles, root_dir, 'nativepython_tests')
    if filterActions:
        testCases = applyFilterActions(filterActions, testCases)

    testCasesToRun += testCases

    if args.list:
        for test in testCasesToRun:
            print(test.id())

        os._exit(0)

    return runPyTestSuite(config, None, testCasesToRun, testArgs)

def executeTests(args, filter_actions):
    if not args.list:
        print("Running python unit tests.")
        print("nose version: ", nose.__version__)
        print(time.ctime(time.time()))

    if runPythonUnitTests(args, filter_actions):
        anyFailed = True
    else:
        anyFailed = False

    print("\n\n\n")

    if anyFailed:
        return 1
    return 0

def main(args):
    #parse args, return zero and exit if help string was printed
    parser = PythonTestArgumentParser()
    args, filter_actions = parser.parse_args(args[1:])

    try:
        return executeTests(args, filter_actions)
    except:
        import traceback
        logging.error("executeTests() threw an exception: \n%s", traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main(sys.argv))
