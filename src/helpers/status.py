#!/usr/bin/env python
# coding: utf-8

"""
A module for displaying status messages on the command line.
"""

from __future__ import division

from datetime import datetime as dt


class Status(object):
    """
    Show a status message encapsulating subtasks.
    
    Create a new instance when the subtask starts, delete it when it finishes.
    A message will be shown in both cases, as well as the (approximate) runtime
    of the respective subtask when it finishes.
    """
    
    verbose = True
    
    def __init__(self, message, verbose=None):
        """
        Create a new instance. Show the passed <message> string if <verbose> is
        True. If <verbose> is None, the <Status.verbose> class variable will be
        evaluated instead (which defaults to True but may be changed).
        
        Remember both the message and the verbosity setting.
        """
        self.__verbose = verbose
        self.__failed = False
        
        verbose = (self.__verbose if self.__verbose is not None
                   else Status.verbose)
        
        if verbose:
            self.__message = message
            print "%s ..." % message
        
        self.__tstart = dt.now()
    
    def fail(self):
        """
        Set the status message on delete to "... fail" rather than "... done".
        """
        self.__failed = True
    
    def update(self, message=None, verbose=None):
        """
        Update the instance with the given <message> and the verbosity level
        given via <verbose>.
        
        If <verbose> is None, the verbosity level will remain unchanged. If
        <message> is None, the message will remain unchanged. Depending on the
        verbosity setting, show the old status message again, as well as the
        (approximate) time that has been passed since the last status update;
        then show the new status message.
        """
        self.__del__()
        message = message if message is not None else self.__message
        verbose = verbose if verbose is not None else self.__verbose
        self.__init__(message, verbose)
    
    def __del__(self):
        """
        On delete, depending on the verbosity setting, show the status message
        again, as well as the (approximate) time that has been passed between
        instance creation and deletion.
        """
        runtime = str(dt.now() - self.__tstart)
        verbose = (self.__verbose if self.__verbose is not None
                   else Status.verbose)
        
        if verbose:
            success = "done" if not self.__failed else "FAILED"
            print "%s ... %s (%s)" % (self.__message, success, runtime)

    
class StatusDecorator(object):
    """
    Show a status message encapsulating methods or functions.
    
    Use instances as method or function decorators: a message will be shown
    when the method or function is entered, another message as well as the
    (approximate) runtime when it is left. The decorator has the following
    syntax
    
        @StatusDecorator(<message>[, <verbose>]),
    
    where <message> is the message string and <verbose> the verbosity (bool),
    which can be used to explicitly overrule the default verbosity.
    
    The default verbosity may be changed on class level via adjusting the
    <StatusDecorator.verbose> variable. This class level setting will then
    apply to all instances except for the ones where the verbosity has been
    explicitly overruled.
    """
    
    verbose = True
    
    def __init__(self, message, verbose=None):
        """
        Initialize, pass the message to be shown and a bool whether to show the
        message (True) or not (False). If <verbose> is None, the
        <StatusDecorator.verbose> class variable will be evaluated instead
        (which defaults to True but may be changed).
        """
        self.__message = message
        self.__verbose = verbose
        
    def __call__(self, f):
        """
        Not to be called manually.
        
        If used as a decorator, the decorated function will be passed as <f>.
        """
        # Two levels of wrapping are necessary, cf. [1].
        #
        # References
        # [1] http://www.artima.com/weblogs/viewpost.jsp?thread=240845 (20120820)
        def wrap_f(*args, **kwargs):
            # Find out whether to use class level or explicit verbosity
            verbose = (self.__verbose if self.__verbose is not None
                       else StatusDecorator.verbose)
            
            # Show the start message if verbose and keep the start time
            if verbose:
                print "%s ..." % self.__message
            self.__tstart = dt.now()
            # Call the actual function
            f_result = f(*args, **kwargs)
            
            # Again, find out verbosity (as class level verbosity may have
            # changed in the meantime)
            verbose = (self.__verbose if self.__verbose is not None
                       else StatusDecorator.verbose)
            
            if verbose:
                # Show the end message as well as the runtime if verbose
                runtime = str(dt.now() - self.__tstart)
                print "%s ... done (%s)" % (self.__message, runtime)
            return f_result
        
        return wrap_f
