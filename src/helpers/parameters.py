#!/usr/bin/env python
# coding: utf-8

from __future__ import division


class TreePrintable(object):
    """
    Pretty-print instances in a tree structure: print the class name (or custom
    <_print_name>, if defined), then successively print all public attributes
    with their values on indented lines. If a value itself is a <TreePrintable>
    instance, indent it further in printing.
    """
    
    @property
    def _print_name(self):
        """
        A custom name to be printed instead of the class name.
        """
        return
    
    @property
    def _print_attr(self):
        """
        A custom set of attributes to be printed instead of the instance
        attributes.
        """
        return
    
    def __repr__(self, prefix="", level=0, blanks=4):
        """
        <prefix>: To be shown before the class name
        <level>: Indentation level
        <blanks>: Number of blanks used per level
        """
        # Assemble the 'name' part of the string
        name = (self.__class__.__name__ if self._print_name is None
                else self._print_name)
        name = (" " * (level * blanks)) + prefix + name
        
        # Assemble the attributes
        try:
            attrs = self._print_attr
        except:
            attrs = None
        if attrs is None:
            attrs = vars(self)
            
        # Add the properties
        all_properties = {}
        for cls in type(self).mro()[::-1]:
        # ^ Invert the method resolution order to have the dictionary entry
        # defined by the lowest class in the hierarchy
            for thingy_name, thingy in cls.__dict__.items():
                if isinstance(thingy, property):
                    try:
                        value = thingy.__get__(self)
                    except AttributeError:
                        value = "[no getter defined]"
                    all_properties["[property] %s" % thingy_name] = value
                    
        # Create lists of tuples (key = attribute name, value = attribute
        # value), hide the private attributes
        attrs_list = [tpl for tpl in attrs.items() if not tpl[0].startswith("_")]
        attrs_list.extend([tpl for tpl in all_properties.items()
                           if not tpl[0].startswith("[property] _")])
        attrs_list = sorted(attrs_list, key = lambda item : item[0].lower())
        
        # Actual attribute strings:
        attr_strings = []
        for k, v in attrs_list:
            try:
                # Try printing as a <TreePrintable>
                attr_string = v.__repr__(prefix="%s: " % k, level=level + 1, blanks=blanks)
            except:
                attr_string = (" " * ((level + 1) * blanks)) + "%s: %s" % (k, v)
            attr_strings.append(attr_string)

        if not attr_strings:
            attr_strings = [(" " * ((level + 1) * blanks)) + "[no attributes]"]

        # Merge and return
        return name + "\n" + "\n".join(attr_strings)


class Parameters(TreePrintable):
    """
    Encapsulate parameters for some function etc. Check if all parameters have
    been set, print all parameters and values in a nice representation.
    
    The parameters are to be represented by instance attributes (i.e.
    <self.my_parameter> or the like).
    """
    
    def check(self, raise_e=False):
        """
        Check if all parameters have been set (i.e. are not None). The values
        themselves are not checked for sanity.
        
        If <raise_e> is True, raise a <RuntimeError> if not all parameters have
        been set. Else, only print a warning and return the unset parameters'
        names as a list.
        """
        unset_params = [k for (k, v) in vars(self).items() if v is None]
        if unset_params:
            msg = "The following parameters have not been set:\n"
            msg += "\n".join(sorted(map(str, unset_params))) 
            if raise_e:
                raise RuntimeError(msg)
            else:
                print msg
                return unset_params
