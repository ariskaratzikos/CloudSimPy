from core.machine import Machine

class Node(object):
    def __init__(self, node_id, topology):
        self.id = node_id

        self.cpu_capacity = 0
        self.memory_capacity = 0
        self.disk_capacity = 0
        self.cpu = 0
        self.memory = 0
        self.disk = 0
        self.topology = topology

        self.cluster = None
        self.machines = []

    def add_machine_capacities(self, machine):
        self.cpu_capacity += machine.cpu_capacity
        self.memory_capacity += machine.memory_capacity
        self.disk_capacity += machine.disk_capacity
        self.cpu += machine.cpu_capacity
        self.memory += machine.memory_capacity
        self.disk += machine.disk_capacity

    def running_task_instances(self, machines=None):
        ls = []
        if machines == None:
            machines = self.machines
        for machine in machines:
            for task_instance in machine.task_instances:
                if task_instance.started and not task_instance.waiting \
                and not task_instance.finished:
                    ls.append(task_instance)
        return ls

    @property
    def finished_task_instances(self):
        ls = []
        for machine in self.machines:
            for task_instance in machine.task_instances:
                if task_instance.finished:
                    ls.append(task_instance)
            return ls

    @property
    def finished_type_task_instances(self):
        service_instances = []
        batch_instances = []
        for machine in self.machines:
            for task_instance in machine.task_instances:
                if task_instance.finished:
                    if task_instance.task.job.type == 2:
                        service_instances.append(task_instance)
                    elif task_instance.task.job.type == 1:
                        batch_instances.append(task_instance)
        return [service_instances, batch_instances]
    
    @property
    def unfinished_task_instances(self):
        ls = []
        for machine in self.machines:
            for task_instance in machine.task_instances:
                if not task_instance.finished:
                    ls.append(task_instance)
        return ls
    
    @property
    def unfinished_service_task_instances(self):
        ls = []
        for machine in self.machines:
            for task_instance in machine.task_instances:
                if not task_instance.finished and task_instance.task.job.type == 2:
                    ls.append(task_instance)
        return ls

    def attach_cluster(self, cluster):
        self.cluster = cluster

    # def attach_machine(self, machine):
    #     self.machines.append(machine)
    #     self.add_machine(machine)

    def add_machines(self, machine_configs):
        for machine_config in machine_configs:
            machine = Machine(machine_config)
            machine.attach_node(self)
            self.machines.append(machine)
            self.add_machine_capacities(machine)

    # def remove_machines(self, machines):
    #     for machine in machines:
    #         machine.dettach_node()
    #         self.machines.remove(machine)

    def delete(self):
        for machine in self.machines:
            self.machines.remove(machine)
            machine.stop_machine()

    def scheduled_time(self, machines=None):  
        avg_time = 0.0
        if machines is None:
            machines = self.machines
        running_task_instances_list = self.running_task_instances(machines)
        for task_instance in running_task_instances_list:
            avg_time += task_instance.response_time + task_instance.running_time
        avg_time = avg_time / len(running_task_instances_list)
        return avg_time

    def service_job_scheduled_time(self, machines=None):
        avg_time = 0.0
        if machines is None:
            machines = self.machines
        running_task_instances_list = self.running_task_instances(machines)
        for task_instance in running_task_instances_list:
            if task_instance.task.job.type == 2:
                avg_time += task_instance.response_time + task_instance.running_time
        avg_time = avg_time / len(running_task_instances_list)
        return avg_time
    
    def response_time(self, machines=None):  
        avg_time = 0.0
        if machines is None:
            machines = self.machines
        running_task_instances_list = self.running_task_instances(machines)
        if len(running_task_instances_list) == 0:
            return 0
        for task_instance in running_task_instances_list:
            avg_time += task_instance.response_time
        avg_time = avg_time / len(running_task_instances_list)
        return avg_time
    
    def finished_response_times(self, machines=None):  
        batch_times = 0
        service_times = 0
        service_len = 0
        batch_len = 0
        if machines is None:
            machines = self.machines
        finished_type_task_instances = self.finished_type_task_instances
        for task_instance in finished_type_task_instances[0]:
            service_times += task_instance.response_time
            service_len += 1
        for task_instance in finished_type_task_instances[1]:
            batch_times += task_instance.response_time
            batch_len += 1
        return [[service_times, service_len], [batch_times, batch_len]]
        
    def remaining_time(self, machines=None):
        avg_time = 0.0
        if machines is None:
            machines = self.machines
        running_task_instances_list = self.running_task_instances(machines)
        for task_instance in running_task_instances_list:
            avg_time += task_instance.duration - task_instance.running_time
        avg_time = avg_time / len(running_task_instances_list)
        return avg_time            

    @property
    def feature(self):
        self.cpu = sum(machine.cpu for machine in self.machines)
        self.memory = sum(machine.memory for machine in self.machines)
        self.disk = sum(machine.disk for machine in self.machines)
        return [self.cpu, self.memory, self.disk]

    def capacity(self, machines=None):
        if machines is None:
            return [self.cpu_capacity, self.memory_capacity, self.disk_capacity]
        else:
            cpu_cap = 0
            mem_cap = 0
            disk_cap = 0
            for machine in machines:
                cpu_cap += machine.cpu_capacity
                mem_cap += machine.memory_capacity
                disk_cap += machine.disk_capacity
            return [cpu_cap, mem_cap, disk_cap]

    def usage(self, machines=None):
        if machines is None:
            machines = self.machines
        capacities = self.capacity(machines)
        cpu = sum(machine.cpu for machine in machines)
        memory = sum(machine.memory for machine in machines)
        disk = sum(machine.disk for machine in machines)
        if machines is None:
            self.cpu = cpu
            self.memory = memory
            self.disk = disk
        return [(cpu / capacities[0]), (memory / capacities[1]), (disk / capacities[2])]

    def avg_usage(self, machines=None):
        if machines is None:
            machines = self.machines
        capacities = self.capacity(machines)
        cpu = sum(machine.cpu for machine in machines)
        memory = sum(machine.memory for machine in machines)
        # disk = sum(machine.disk for machine in machines)
        return (((cpu / capacities[0]) + (memory / capacities[1])) / 2)
        # return ((cpu / capacities[0]) + (memory / capacities[1]) + (disk / capacities[2])) / 3

    def avg_batch_usage(self,machines=None):
        if machines is None:
            machines = self.machines
        capacities = self.capacity(machines)
        running_task_instances_list = self.running_task_instances(machines)
        for task_instance in running_task_instances_list:
            if task_instance.type == 1:
                temp_cpu -= task_instance.cpu
                temp_mem -= task_instance.memory
                temp_disk -= task_instance.disk
        return ((temp_cpu / capacities[0]) + (temp_mem / capacities[1]) + (temp_disk / capacities[2])) / 3

    def running_batch_task_instances(self, machines=None):
        ls = []
        if machines is None:
            machines = self.machines       
        for machine in machines:
            for task_instance in machine.task_instances:
                if task_instance.task.job.type == 1 and task_instance.started and not task_instance.finished:
                    ls.append(task_instance)
        return ls

    def running_service_task_instances(self, machines=None):
        ls = []
        if machines is None:
            machines = self.machines       
        for machine in machines:
            for task_instance in machine.task_instances:
                if task_instance.task.job.type == 2 and task_instance.started and not task_instance.finished:
                    ls.append(task_instance)
        return ls

    def waiting_task_instances(self, machines=None):
        ls = []
        if machines is None:
            machines = self.machines       
        for machine in machines:
            for task_instance in machine.task_instances:
                if (task_instance.waiting and not task_instance.finished):
                    ls.append(task_instance)
        return ls

    def non_waiting_instances(self, machines=None):
        ls = []
        if machines is None:
            machines = self.machines       
        for machine in machines:
            for task_instance in machine.task_instances:
                if (task_instance.running and not task_instance.waiting):
                    ls.append(task_instance)
        return ls 

    def workload_accomodation(self, workload):
        ls = []
        for machine in self.machines:
            if machine.accommodate(workload):
                ls.append(machine)
        return ls
